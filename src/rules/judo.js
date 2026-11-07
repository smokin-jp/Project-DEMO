// JUDO ruleset (IJF contest rules, 2025 update as implemented by national federations).
// Sources: British Judo summary of the IJF 2025 rule changes (britishjudo.org.uk, 22 Feb 2025):
//   yuko reinstated; waza-ari for >90 degree landings toward the back; hold-down 5-9 s yuko;
//   non-combativity time 30 s; grips below inner-thigh used negatively = shido; kansetsu-waza throws:
//   shido (low risk) / hansoku-make (uke cannot escape); head throws no longer HM for junior/senior.
// Ippon/waza-ari/osaekomi timings follow the IJF refereeing rules (ippon 20 s, waza-ari 10-19 s, yuko 5-9 s).
// NOT verified from the primary rulebook: the shido count (3rd shido = hansoku-make here) and whether
// waza-ari awasete ippon is active at your event: both are config fields below.

const fouls = {
  false_attack: { label: 'False attack', desc: 'Attack with no realistic chance of throwing (2025: includes no real possibility).' },
  passivity: { auto: true, label: 'Non-combativity (30 s)', desc: 'No attack for 30 s. Ne-waza actions can be taken into account.' },
  stepping_out: { label: 'Leaving the contest area (engaged in no action)', desc: 'Unintentional stepping out is no longer shido; stepping out when not engaged in a judo action is.' },
  illegal_grip: { label: 'Illegal grip / grips below the inner thigh', desc: 'Pistol grip, clasped hands (bear hug) and negative grips below the inner thigh.' },
  leg_grab: { label: 'Gripping / blocking legs below the belt', auto: true, desc: 'Hooking, blocking or gripping below the inner-thigh level. AI call from pose geometry (a hand kept on the opponent\'s leg for 0.7 s): check it.' },
  defensive: { label: 'Excessively defensive posture', desc: 'Bent over / blocking without attacking.' },
  strike: { label: 'Striking the opponent', direct: 'hm', desc: 'Hitting with fist or elbow is a serious infringement: hansoku-make. Referee call only: the AI flags a possible strike but never disqualifies on its own (grip fighting looks like punching to a camera).' },
  kicking: { label: 'Kicking / kneeing the opponent', auto: true, direct: 'hm', desc: 'Kicks and knee strikes are prohibited: hansoku-make.' },
  dangerous_throw: { label: 'Dangerous technique (e.g. kawazu-gake, high-risk kansetsu throw)', direct: 'hm', desc: 'Direct hansoku-make.' },
  disregard: { label: 'Disregarding referee / unsportsmanlike', direct: 'hm', desc: 'Direct hansoku-make.' },
};

// A "kick" that lands on the opponent's legs is a sweep or reap (de-ashi-barai, osoto-gari ...), which are legal techniques
const SWEEP_ZONES = new Set(['legs', 'knee_joint']);
// a leg action is a throw attempt if the opponent goes down within this long before or after it is judged
const FALL_LINK_MS = 3000;

export default {
  id: 'judo',
  name: 'Judo',
  icon: '🥋',
  defaultVariant: 'senior',
  variants: {
    senior: { label: 'IJF 2025 – Senior (4:00 + Golden Score)', rounds: 1, roundSec: 240, restSec: 0, ladder: ['shido', 'shido', 'hm'], awasete: true, passivityMs: 30000 },
    demo: { label: 'DEMO clock (1:30 + Golden Score)', rounds: 1, roundSec: 90, restSec: 0, ladder: ['shido', 'shido', 'hm'], awasete: true, passivityMs: 30000 },
  },
  fouls,
  manualScores: [
    { kind: 'ippon', label: 'Ippon' },
    { kind: 'waza-ari', label: 'Waza-ari' },
    { kind: 'yuko', label: 'Yuko' },
    { kind: 'hold_yuko', label: 'Osaekomi 5–9 s (yuko)' },
    { kind: 'hold_wazaari', label: 'Osaekomi 10–19 s (waza-ari)' },
    { kind: 'hold_ippon', label: 'Osaekomi 20 s (ippon)' },
  ],
  summary: [
    { title: 'Scores', items: [
      'IPPON (match ends): thrown largely on the back with considerable force and speed; or a hold-down for 20 s.',
      'WAZA-ARI: throw landing toward the back (>90°) without full ippon force; hold-down 10–19 s. Two waza-ari = waza-ari awasete ippon (match ends).',
      'YUKO (reinstated 2025): landing on side / buttocks; hold-down 5–9 s. Ranks below waza-ari.',
      'The AI grades the Osoto-gari landing from torso angle, fall speed and fall duration: ippon / waza-ari / yuko.',
    ] },
    { title: 'Penalties', items: [
      'SHIDO: false attack, non-combativity (30 s), illegal grips, grips below the inner thigh, excessively defensive posture, stepping out when not engaged. Shidos give no points in 2025.',
      '3rd shido = HANSOKU-MAKE (disqualification), opponent wins.',
      'Direct HANSOKU-MAKE: striking/kicking, dangerous technique (e.g. kawazu-gake, high-risk kansetsu throw), disregarding the referee.',
    ] },
    { title: 'Time', items: ['4:00 contest. If level (no score decision) → Golden Score, first score wins; no clock limit.', 'Osaekomi 5 s ends golden score as yuko.'] },
  ],
  sources: [
    { title: 'British Judo: IJF 2025 contest rules', url: 'https://www.britishjudo.org.uk/2025-contest-rules-to-be-implemented-from-22nd-february/' },
    { title: 'IJF refereeing rules (2025-2028)', url: 'https://www.ijf.org/' },
  ],
  verify: ['Shido count (3 → hansoku-make) and waza-ari awasete ippon should be checked against the IJF 2025-2028 rulebook.'],

  sideDefaults: () => ({ wazaari: 0, yuko: 0, shido: 0, ippon: 0 }),
  init(m) {
    // non-combativity (passivity): ms since each fighter last attacked, counted only while both are standing and in view
    m.sinceAttack = { a: 0, b: 0 };
    m.lastTrack = -1e9;
    m.trackedOnce = false;
    m.fallAt = { a: -1e9, b: -1e9 }; // when each fighter last went down (match time)
  },

  // time-based rule, called by the engine every tick while the bout is live
  onTick(m, dt) {
    const limit = m.cfg.passivityMs;
    if (!limit || !m.trackedOnce) return;
    if (m.now - m.lastTrack > 1500) return; // ground work / nobody in view: the standing-contest timer pauses
    for (const side of ['a', 'b']) {
      m.sinceAttack[side] += dt;
      if (m.sinceAttack[side] >= limit) {
        m.sinceAttack[side] = 0; // the next 30 s start now
        m.penalize(side, 'passivity', { auto: true });
        if (m.status === 'ended') return;
      }
    }
  },

  handle(ev, m) {
    const S = ev.by ? m.sides[ev.by] : null;
    const auto = !!ev.auto;
    switch (ev.type) {
      case 'tracking': // both fighters in view and on their feet
        m.lastTrack = m.now;
        m.trackedOnce = true;
        break;
      case 'attack': // a step in or a leg swing towards the opponent counts as engaging
        m.sinceAttack[ev.by] = 0;
        break;
      case 'throw':
        m.sinceAttack.a = m.sinceAttack.b = 0;
        m.fallAt[ev.against ?? m.other(ev.by)] = m.now;
        this.award(m, ev.by, ev.grade, ev.technique, auto, ev);
        break;
      case 'score':
        if (ev.kind === 'hold_yuko') this.award(m, ev.by, 'yuko', 'Osaekomi', false);
        else if (ev.kind === 'hold_wazaari') this.award(m, ev.by, 'waza-ari', 'Osaekomi', false);
        else if (ev.kind === 'hold_ippon') this.award(m, ev.by, 'ippon', 'Osaekomi', false);
        else this.award(m, ev.by, ev.kind, 'Manual', false);
        break;
      case 'strike_landed': // backend offline: geometry-only candidate
        this.possibleStrike(m, ev, 'a hand moving fast towards the opponent');
        break;
      case 'down': // a fall: a leg action that ends with the opponent on the mat is a throw attempt, not a kick
        m.fallAt[ev.by] = m.now;
        break;
      case 'strike_verdict': {
        const v = ev.verdict;
        if ((v.strike || 'punch') !== 'punch') {
          // a kick / knee the backend confirms is real AND that touched the opponent somewhere a sweep or reap would not
          if (v.strike_check?.value && v.contact.value && !SWEEP_ZONES.has(v.contact_location.value)) this.kickCall(m, ev);
          else if (v.strike_check?.value && v.contact.value) m.say(`Leg action on the opponent's legs (${v.contact_location.value.replace('_', ' ')}): a foot sweep / reap, not a kick`, { kind: 'info', side: ev.by, auto: true });
        } else if (v.contact.value && v.genuine_punch.detail.classification !== 'miss') {
          this.possibleStrike(m, ev, `the video judge saw contact (${v.genuine_punch.detail.classification})`);
        }
        break;
      }
      case 'kick_landed': // backend offline: geometry-only fallback
        if (ev.target !== 'leg') this.kickCall(m, ev);
        break;
      case 'leg_grab': // pose geometry: a hand kept on the opponent's leg below the belt
        if (m.status === 'running') m.penalize(ev.by, 'leg_grab', ev);
        break;
      case 'foul':
        m.penalize(ev.by, ev.foul, ev);
        break;
      default:
    }
  },

  /**
   * A hand moving fast at the opponent. In judo that is almost always grip fighting, a push or a throw entry, and a camera cannot
   * tell it from a punch: a wrong automatic call here ends the bout (hansoku-make), which tested far too often on real throw clips.
   * So the AI only FLAGS it; the referee issues the hansoku-make from the override panel if it really was a strike.
   */
  possibleStrike(m, ev, why) {
    if (m.status !== 'running') return;
    m.say(`Possible strike by ${m.sides[ev.by].name} (${why}): check the replay. Not penalised; use the override panel if it was a real strike`, { kind: 'info', side: ev.by, auto: true, evidence: ev });
  },

  /**
   * A kick or knee that touched the opponent's body or head. Osoto-gari, Ouchi-gari, Harai-goshi, Uchi-mata and foot sweeps all
   * swing a leg at the opponent and look like a kick to a camera, and the fall they cause usually arrives AFTER the clip
   * is judged. So the call waits a moment: if the opponent goes down shortly before or after, it was a technique.
   */
  kickCall(m, ev) {
    if (m.status !== 'running') return;
    const against = ev.against ?? m.other(ev.by);
    if (m.now - m.fallAt[against] < FALL_LINK_MS) {
      m.say(`Leg action followed by ${m.sides[against].name} going down: judged a throw attempt, not a kick`, { kind: 'info', side: ev.by, auto: true });
      return;
    }
    const t0 = m.now;
    m.later(FALL_LINK_MS, () => {
      if (m.status !== 'running') return;
      if (m.fallAt[against] >= t0) {
        m.say(`Leg action followed by ${m.sides[against].name} going down: judged a throw attempt, not a kick`, { kind: 'info', side: ev.by, auto: true });
        return;
      }
      m.curAuto = true;
      try { m.penalize(ev.by, 'kicking', ev); } finally { m.curAuto = false; }
    });
  },

  award(m, by, grade, tech, auto, ev) {
    const S = m.sides[by];
    const o = m.sides[m.other(by)];
    const t = tech && tech !== 'Manual' ? ` (${tech})` : '';
    if (grade === 'ippon') {
      m.end(by, 'IPPON', `IPPON${t}! ${S.name} wins`);
      return;
    }
    if (grade === 'waza-ari') {
      S.wazaari++;
      if (S.wazaari >= 2 && m.cfg.awasete) {
        m.end(by, 'IPPON', `WAZA-ARI AWASETE IPPON${t}! ${S.name} wins`, `Waza-ari awasete ippon. ${S.name} wins`);
        return;
      }
      m.say(`WAZA-ARI${t} to ${S.name}`, { kind: 'score', side: by, big: 'WAZA-ARI', auto, evidence: ev });
    } else if (grade === 'yuko') {
      S.yuko++;
      m.say(`YUKO${t} to ${S.name}`, { kind: 'score', side: by, big: 'YUKO', auto, evidence: ev });
    }
    if (m.golden) m.end(by, 'GOLDEN SCORE', `Golden Score: ${S.name} scores and wins`);
    void o;
  },

  applyPenalty(action, side, foulId, f, m) {
    const S = m.sides[side];
    const o = m.other(side);
    if (action === 'hm') {
      m.end(o, 'HANSOKU-MAKE', `HANSOKU-MAKE for ${S.name}: ${f.label}. ${m.sides[o].name} wins`, `Hansoku make. ${m.sides[o].name} wins`);
    } else {
      S.shido++;
      m.say(`SHIDO ${S.shido} to ${S.name}: ${f.label}`, { kind: 'foul', side, big: `SHIDO ${S.shido}`, speak: `Shido, ${S.name}` });
    }
  },

  onTimeUp(m) {
    const A = m.sides.a, B = m.sides.b;
    m.say('Time!', { big: 'SONO-MAMA' });
    const ka = [A.wazaari, A.yuko], kb = [B.wazaari, B.yuko];
    if (ka[0] !== kb[0]) m.end(ka[0] > kb[0] ? 'a' : 'b', 'DECISION', `${ka[0] > kb[0] ? A.name : B.name} wins on waza-ari`);
    else if (ka[1] !== kb[1]) m.end(ka[1] > kb[1] ? 'a' : 'b', 'DECISION', `${ka[1] > kb[1] ? A.name : B.name} wins on yuko`);
    else {
      m.golden = true;
      m.clock = 0;
      m.say('Level: GOLDEN SCORE, next score wins', { big: 'GOLDEN SCORE' });
    }
  },
};
