// BOXING ruleset.
// Sources used: ABC Unified Rules of Boxing (abcboxing.com/unified-rules-boxing: 3-min rounds, 10-point
// must system, mandatory 8-count, no standing 8 / no 3-knockdown rule, 2-pt deduction for intentional foul with
// injury, DQ for flagrant fouls) and amateur (World Boxing / IBA style) practice: caution -> warnings, 3 warnings = DQ,
// standing 8-count and stoppage at 3 knockdowns. Check your tournament's own rulebook before relying on any number.

const fouls = {
  low_blow: { label: 'Low blow (below the belt)', auto: true, desc: 'Punch landing below the belt line.' },
  hit_after_bell: { label: 'Hitting after the bell / on the break', auto: true, desc: 'Strike thrown while the round is not live.' },
  holding: { label: 'Holding / clinching', auto: true, desc: 'Sustained clinch without working.' },
  wrestling: { label: 'Wrestling / throwing', auto: true, desc: 'Throwing, tripping or wrestling the opponent down.' },
  headbutt: { label: 'Head-butt', auto: true, desc: 'Intentional use of the head. AI call from pose geometry (head thrust that meets the opponent\'s head): check it.' },
  elbow: { auto: true, label: 'Elbow / forearm / shoulder', desc: 'Striking with anything but the knuckle part of the glove.' },
  open_glove: { auto: true, label: 'Open glove / palm / wrist', desc: 'Hitting with the open glove, inside or wrist.' },
  back_head: { auto: true, label: 'Back of head / neck / kidney', desc: 'Rabbit punch or kidney punch.' },
  kicking: { label: 'Kicking / kneeing', auto: true, direct: 'dq', desc: 'Kicks and knee strikes are illegal in boxing: disqualification.' },
  hitting_down: { auto: true, label: 'Hitting a downed fighter', desc: 'Striking an opponent who is down or getting up.' },
  ducking_low: { label: 'Ducking below the belt', auto: true, desc: 'Dangerously ducking below the waist. AI call from pose geometry (head held below the opponent\'s belt line).' },
  mouthguard: { label: 'Spitting out mouthguard', desc: 'Deliberately spitting out the mouthpiece.' },
  flagrant: { label: 'FLAGRANT / intentional foul with injury', direct: 'dq', desc: 'Intentional foul severe enough to end the bout.' },
  unsportsmanlike: { label: 'Unsportsmanlike conduct', direct: 'dq', desc: 'Referee may stop the bout and disqualify.' },
};

export default {
  id: 'boxing',
  name: 'Boxing',
  icon: '🥊',
  defaultVariant: 'amateur',
  variants: {
    amateur: { label: 'Amateur (3 × 3:00, 3 warnings = DQ)', rounds: 3, roundSec: 180, restSec: 60, ladder: ['caution', 'warning', 'warning', 'dq'], kdStop: 3 },
    pro: { label: 'Professional – Unified Rules (4 rds shown)', rounds: 4, roundSec: 180, restSec: 60, ladder: ['warning', 'deduct', 'deduct', 'dq'], kdStop: 99, drawsAllowed: true },
    demo: { label: 'DEMO clock (2 × 0:45)', rounds: 2, roundSec: 45, restSec: 8, ladder: ['caution', 'warning', 'warning', 'dq'], kdStop: 3 },
  },
  fouls,
  manualScores: [
    { kind: 'punch', label: 'Award clean punch' },
    { kind: 'knockdown', label: 'Knockdown' },
    { kind: 'stop', label: 'Referee stops fight (TKO)' },
  ],
  summary: [
    { title: 'How a bout is scored', items: [
      '10-Point Must system per round: round winner 10, loser 9; 10-8 for clear dominance or a knockdown; 10-10 even.',
      'The AI scores clean punches landed on the head / front-or-side of the torso above the belt (no points for blocked or low punches).',
      'Each knockdown costs the downed fighter 1 point; each warning / deduction costs the offender 1 point.',
      'Judges total all rounds; most points wins. Unified pro: a tie is a draw.',
    ] },
    { title: 'Knockdowns', items: [
      'Mandatory 8-count after every knockdown: the fighter may not continue before 8 even if upright earlier.',
      'Not up by 10 = KO. Amateur: 3 knockdowns in a round stops the bout. Unified pro: no 3-knockdown rule, no standing 8-count.',
      'A slip without a preceding fair blow is NOT a knockdown (AI checks for a landed punch in the previous 3 s).',
    ] },
    { title: 'Fouls and penalties', items: [
      'Illegal: low blows, holding, hitting after the bell or on the break, wrestling/throwing, head-butts, elbows, open glove, rabbit/kidney punches, kicking, hitting a downed fighter, ducking below the belt.',
      'Amateur ladder: caution → warning (−1) → warning (−1) → 3rd warning = disqualification.',
      'Pro ladder: warning → −1 point → −1 point → DQ. Intentional foul that ends the fight = DQ; accidental = no contest / technical decision.',
      'Flagrant fouls and unsportsmanlike conduct can be disqualified immediately.',
    ] },
    { title: 'How a bout ends', items: ['KO (not up by 10)', 'TKO / RSC (referee stops it; 3 knockdowns in amateur)', 'DQ (fouls)', 'Decision on points after the final round'] },
  ],
  sources: [
    { title: 'ABC Unified Rules of Boxing', url: 'https://www.abcboxing.com/unified-rules-boxing' },
  ],
  verify: ['Amateur warning ladder and 3-knockdown stoppage are from World Boxing/IBA practice: confirm against your event rulebook.'],

  sideDefaults: () => ({ score: 0, punches: 0, punchesTotal: 0, kdRound: 0, kdTotal: 0, deductRound: 0, deductTotal: 0, roundScores: [] }),

  init(m) {
    m.lastLanded = { a: -1e9, b: -1e9 };
    m.count = null;
    m.roundCards = []; // one entry per finished round: the 10-point-must card and what produced it
    m.downAt = { a: null, b: null }; // detection-clock time at which each fighter went down (null = on his feet)
  },

  /**
   * A strike that lands on a fighter who is already on the mat is a foul ("striking a downed fighter").
   * Compare WHEN the strike happened (its peak time) with when the fighter went down: the punch that caused the knockdown
   * is judged a moment later, after he is already down, and must not be penalised.
   */
  hitDown(m, ev) {
    const v = ev.verdict;
    const down = m.downAt?.[ev.against];
    const t = ev.peakT ?? ev.clock;
    if (down == null || t == null || t <= down + 150) return false;
    const touched = v ? v.contact.value && (!v.strike_check || v.strike_check.value) : true; // geometry fallback events are "landed" already
    if (!touched) return false;
    m.penalize(ev.by, 'hitting_down', ev);
    return true;
  },

  handle(ev, m) {
    const by = ev.by;
    const S = by ? m.sides[by] : null;
    const auto = !!ev.auto;
    switch (ev.type) {
      case 'strike_landed': {
        if (m.status !== 'rest' && m.status !== 'paused' && this.hitDown(m, ev)) return;
        if (m.status === 'running') {
          if (ev.target === 'low') {
            m.penalize(by, 'low_blow', ev);
            return;
          }
          S.punches++;
          S.punchesTotal++;
          m.lastLanded[ev.against] = m.now;
          m.say(`${S.name} lands a clean punch to the ${ev.target}${ev.virtual ? ' (virtual opponent)' : ''}  [${S.punches} this round]`, {
            kind: 'score', side: by, big: 'PUNCH LANDS', speak: `Clean punch. ${S.name}`, auto, evidence: ev,
          });
        } else if (m.status === 'rest' || m.status === 'paused') {
          m.penalize(by, 'hit_after_bell', ev);
        }
        break;
      }
      case 'strike_verdict': {
        // Verdict from the CLIP backend (answers the 7 scoring questions). See backend/app/judge.py
        const v = ev.verdict;
        if ((v.strike || 'punch') !== 'punch') {
          // a kick or knee strike is illegal in boxing: disqualification (once the backend confirms it is a real strike, not a step)
          if (v.strike_check?.value) m.penalize(by, 'kicking', ev);
          else m.say(`Ignored: not a real ${v.strike} (step / raised leg)`, { kind: 'info', side: by, auto });
          break;
        }
        if (m.status !== 'rest' && m.status !== 'paused' && this.hitDown(m, ev)) break; // striking a downed fighter
        const loc = String(v.contact_location.value).replace(/_/g, ' ');
        const pct = Math.round(v.confidence * 100);
        if (m.status === 'rest' || m.status === 'paused') {
          if (v.contact.value) m.penalize(by, 'hit_after_bell', ev);
          break;
        }
        if (m.status !== 'running') break;
        if (v.foul) {
          m.penalize(by, v.foul, ev);
        } else if (v.scored) {
          this.awardPunch(m, by, ev.against, `${loc}${v.intended_target.value ? '' : ''}`, auto, `${pct}% confidence`, ev);
        } else if (v.review) {
          m.say(`REVIEW: ${v.summary}`, { kind: 'info', side: by, big: 'REVIEW', speak: 'Referee review', auto, evidence: ev });
        } else if (v.contact.value) {
          m.say(`No score: ${v.summary}`, { kind: 'info', side: by, auto, evidence: ev });
        }
        break;
      }
      case 'kick_landed': // backend offline: geometry-only fallback
        m.penalize(by, 'kicking', ev);
        break;
      case 'throw':
        if (m.status === 'running') m.penalize(by, 'wrestling', ev);
        break;
      case 'holding':
        if (m.status === 'running') m.penalize(by, 'holding', ev);
        break;
      case 'down':
        m.downAt[ev.by] = ev.clock ?? m.now; // on the mat, however he got there (knockdown, slip or throw)
        if (ev.thrown || m.status !== 'running') break;
        if (m.now - m.lastLanded[ev.by] < 3000) this.startCount(m, ev.by, auto);
        else m.say(`${S.name} is down but there was no preceding blow: slip, no count`, { kind: 'info', side: ev.by, auto });
        break;
      case 'up':
        m.downAt[ev.by] = null;
        if (m.status === 'counting' && m.count.side === ev.by) {
          m.count.up = true;
          m.say(`${S.name} is up. Mandatory 8-count continues`, { kind: 'info', side: ev.by, auto });
        }
        break;
      // pose-geometry fouls (src/detect/foul.js). A sport without the matching foul ignores the event.
      case 'headbutt':
        if (m.status === 'running') m.penalize(by, 'headbutt', ev);
        break;
      case 'low_head':
        if (m.status === 'running') m.penalize(by, 'ducking_low', ev);
        break;
      case 'leg_hold': // a leg held for seconds (kickboxing); boxing has no such foul
        if (m.status === 'running') m.penalize(by, 'leg_hold', ev);
        break;
      case 'foul':
        m.penalize(ev.by, ev.foul, ev);
        break;
      case 'score':
        if (ev.kind === 'punch') this.awardPunch(m, ev.by, m.other(ev.by), 'body/head', false, 'referee award', ev);
        else if (ev.kind === 'knockdown') this.startCount(m, ev.by, false);
        else if (ev.kind === 'stop') m.end(m.other(ev.by), 'TKO', `Referee stops the contest. ${m.sides[m.other(ev.by)].name} wins by TKO`);
        break;
      default:
    }
  },

  awardPunch(m, by, against, where, auto, why, ev, what = 'punch') {
    const S = m.sides[by];
    S.punches++;
    S.punchesTotal++;
    m.lastLanded[against] = m.now;
    m.say(`${S.name} lands a clean ${what} to the ${where}  [${S.punches} this round · ${why}]`, {
      kind: 'score', side: by, big: `${what.toUpperCase()} LANDS`, speak: `Clean ${what}. ${S.name}`, auto, evidence: ev,
    });
  },

  startCount(m, side, auto) {
    const S = m.sides[side];
    S.kdRound++;
    S.kdTotal++;
    m.status = 'counting';
    m.count = { side, n: 0, up: false, said: 0 };
    m.say(`KNOCKDOWN! ${S.name} is down. Mandatory 8-count`, { kind: 'score', side: m.other(side), big: 'KNOCKDOWN', auto });
    if (S.kdRound >= m.cfg.kdStop) m.end(m.other(side), 'RSC', `Three knockdowns in the round: contest stopped. ${m.sides[m.other(side)].name} wins`);
  },

  onCountTick(m, dt) {
    const c = m.count;
    c.n += dt / 1000;
    const whole = Math.floor(c.n);
    if (whole > c.said && whole <= 10) {
      c.said = whole;
      m.say(String(whole), { kind: 'count', side: c.side, speak: String(whole) });
    }
    if (c.up && c.n >= 8) {
      m.status = 'running';
      m.count = null;
      m.say('Box on!', { kind: 'info', big: 'BOX' });
    } else if (c.n >= 10) {
      m.end(m.other(c.side), 'KO', `${m.sides[c.side].name} did not beat the count. ${m.sides[m.other(c.side)].name} wins by KO`);
    }
  },

  applyPenalty(action, side, foulId, f, m) {
    const S = m.sides[side];
    const label = f.label;
    const o = m.other(side);
    if (action === 'dq') {
      m.end(o, 'DQ', `${S.name} DISQUALIFIED for ${label.toLowerCase()}. ${m.sides[o].name} wins`);
    } else if (action === 'caution') {
      m.say(`CAUTION to ${S.name}: ${label}`, { kind: 'foul', side, big: 'CAUTION' });
    } else {
      S.deductRound++;
      S.deductTotal++;
      m.say(`${action === 'warning' ? 'WARNING' : 'POINT DEDUCTION'} to ${S.name}: ${label} (−1 point)`, { kind: 'foul', side, big: action === 'warning' ? 'WARNING' : 'POINT DEDUCTED' });
    }
  },

  scoreRound(m) {
    const A = m.sides.a, B = m.sides.b;
    let ra = 10, rb = 10;
    if (A.punches > B.punches) {
      rb = A.punches >= 6 && A.punches >= 3 * Math.max(B.punches, 1) ? 8 : 9;
    } else if (B.punches > A.punches) {
      ra = B.punches >= 6 && B.punches >= 3 * Math.max(A.punches, 1) ? 8 : 9;
    }
    ra -= A.kdRound + A.deductRound;
    rb -= B.kdRound + B.deductRound;
    // a knockdown / deduction must not leave the offender ahead
    A.score += ra;
    B.score += rb;
    A.roundScores.push(ra);
    B.roundScores.push(rb);
    const card = (S, pts) => ({ pts, strikes: S.punches, kd: S.kdRound, ded: S.deductRound });
    m.roundCards.push({ round: m.round, a: card(A, ra), b: card(B, rb) });
    m.say(`Round ${m.round} scores: RED ${ra} – BLUE ${rb}  (punches ${A.punches}–${B.punches})`, { kind: 'info' });
    for (const S of [A, B]) {
      S.punches = 0;
      S.kdRound = 0;
      S.deductRound = 0;
    }
  },

  onTimeUp(m) {
    this.scoreRound(m);
    if (m.round >= m.cfg.rounds) {
      const A = m.sides.a, B = m.sides.b;
      m.say('Final bell', { big: 'BELL' });
      if (A.score > B.score) m.end('a', 'DECISION', `${A.name} wins on points ${A.score}–${B.score}`);
      else if (B.score > A.score) m.end('b', 'DECISION', `${B.name} wins on points ${B.score}–${A.score}`);
      else if (m.cfg.drawsAllowed) m.end(null, 'DRAW', `Draw ${A.score}–${B.score}`);
      else {
        const wa = A.deductTotal, wb = B.deductTotal;
        if (wa !== wb) m.end(wa < wb ? 'a' : 'b', 'DECISION', `Level on points: fewer warnings wins (${wa < wb ? A.name : B.name})`);
        else m.end(null, 'DRAW', 'Level on points: judges decide');
      }
    } else {
      m.status = 'rest';
      m.clock = m.cfg.restSec * 1000;
      m.say(`End of round ${m.round}`, { big: 'BELL' });
    }
  },

  onRestOver(m) {
    m.round++;
    m.status = 'running';
    m.clock = m.cfg.roundSec * 1000;
    m.say(`Seconds out: round ${m.round}`, { big: `ROUND ${m.round}` });
  },
};
