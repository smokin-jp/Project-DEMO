// KICKBOXING ruleset. Same bout mechanics as boxing (rounds, 10-point-must, mandatory 8-count, knockdown stoppage,
// foul ladder, DQ) so it extends the boxing module, with kickboxing's legal strikes, targets and fouls.
//
// VERIFIED from the ABC Unified Rules of Professional Kickboxing (abcboxing.com/?p=1487):
//   up to 10 x 3:00 rounds, 10-point must (10-9/10-8/10-7), scoring priority = knockdowns > cumulative impact >
//   clean strikes > ring generalship; legal: boxing punches, spinning backfists, knees, kicks above the belt and
//   foot/shin kicks to the legs (not thrusting kicks to the knee joint); targets = front/sides of head, front/sides of
//   torso above the belt, the leg (not the knee joint); mandatory 8-count, no standing 8, 3 knockdowns from head
//   strikes or kicks in a round = TKO; fouls list below; referee may deduct points or disqualify.
// PARTLY VERIFIED (WAKO amateur, wako.sport rulebooks not read in full): Full Contact = strikes above the waist, no knees,
//   elbows, clinching or throws; Low Kick = Full Contact plus kicks to the thighs; 3 x 2:00 rounds. WAKO penalty ladder,
//   knee legality in Low Kick and the 3-knockdown rule are NOT confirmed: edit the variant config to your event.
//
// The pose detector triggers on punches, kicks and knees; the CLIP backend confirms each one and judges contact, target and
// legality (legal target zones per variant). The referee-override buttons remain for anything the AI cannot see.
import boxing from './boxing.js';

const fouls = {
  low_blow: { label: 'Groin / low blow', auto: true, desc: 'Punch landing below the belt line (groin).' },
  hit_after_bell: { label: 'Striking after the bell / on the break', auto: true, desc: 'Strike thrown while the round is not live.' },
  holding: { label: 'Holding / clinching without working', auto: true, desc: 'Clinching is allowed to launch a knee; referee separates after ~5 s.' },
  wrestling: { label: 'Throwing / sweeping / takedown', auto: true, desc: 'Downing the opponent by anything but legal strikes (throws, sweeps, reaps, takedowns, joint locks).' },
  kick_below_waist: { auto: true, label: 'Kick below the waist (Full Contact)', desc: 'Leg kicks are illegal in WAKO Full Contact; legal in Low Kick and pro.' },
  illegal_knee: { auto: true, label: 'Illegal knee strike', desc: 'Knees are not allowed in WAKO Full Contact.' },
  knee_joint_kick: { auto: true, label: 'Thrusting / linear kick to the knee joint', desc: 'Foul in pro rules.' },
  headbutt: { label: 'Head-butt', auto: true, desc: 'Intentional use of the head. AI call from pose geometry (head thrust that meets the opponent\'s head): check it.' },
  elbow: { auto: true, label: 'Elbow / forearm / non-padded part of glove', desc: 'Strikes with the elbow, arm, wrist or non-padded glove.' },
  open_glove: { auto: true, label: 'Open glove / palm', desc: 'Hitting with the open glove.' },
  back_head: { auto: true, label: 'Back of head / spine', desc: 'Rabbit punch or strike to the spine.' },
  throat: { label: 'Attack to the throat', desc: 'Strike to the throat.' },
  hitting_down: { auto: true, label: 'Striking a downed fighter', desc: 'Striking an opponent who is down.' },
  leg_hold: { label: 'Holding the opponent’s leg', auto: true, desc: 'Holding a kicked leg without a legal strike, or throwing more than one technique while holding it.' },
  ropes: { label: 'Holding / using the ropes', desc: 'Holding or using the ropes while striking or clinching.' },
  timidity: { label: 'Timidity / avoiding contact', desc: 'Repeatedly and intentionally avoiding contact.' },
  delay: { label: 'Intentional delay', desc: 'Repeatedly spitting out the mouthpiece, deliberately falling out of the ring.' },
  disobey: { label: 'Failing to obey referee commands', desc: 'Ignoring "break" or "stop".' },
  thumbing: { label: 'Thumbing / biting / spitting', desc: 'Eye-thumbing, biting or spitting at the opponent.' },
  abusive: { label: 'Abusive language or gestures', desc: 'Unsportsmanlike words or gestures.' },
  flagrant: { label: 'FLAGRANT / intentional foul with injury', direct: 'dq', desc: 'Intentional foul severe enough to end the bout.' },
  unsportsmanlike: { label: 'Unsportsmanlike conduct', direct: 'dq', desc: 'Referee may stop the bout and disqualify.' },
};

const KICKS = { kick: 'kick', knee: 'knee strike', leg_kick: 'leg kick' };

export default {
  ...boxing,
  id: 'kickboxing',
  name: 'Kickboxing',
  icon: '🦵',
  defaultVariant: 'unified',
  variants: {
    unified: { label: 'Pro – Unified Rules (3 × 3:00 shown)', rounds: 3, roundSec: 180, restSec: 60, ladder: ['caution', 'deduct', 'deduct', 'dq'], kdStop: 3, drawsAllowed: true, legKicks: true, knees: true, legalZones: ['head', 'torso', 'legs'] },
    wako_fc: { label: 'WAKO Full Contact (3 × 2:00, above the waist)', rounds: 3, roundSec: 120, restSec: 60, ladder: ['caution', 'warning', 'deduct', 'dq'], kdStop: 3, drawsAllowed: false, legKicks: false, knees: false, legalZones: ['head', 'torso'] },
    wako_lk: { label: 'WAKO Low Kick (3 × 2:00, leg kicks legal)', rounds: 3, roundSec: 120, restSec: 60, ladder: ['caution', 'warning', 'deduct', 'dq'], kdStop: 3, drawsAllowed: false, legKicks: true, knees: true, legalZones: ['head', 'torso', 'legs'] },
    demo: { label: 'DEMO clock (2 × 0:45)', rounds: 2, roundSec: 45, restSec: 8, ladder: ['caution', 'deduct', 'deduct', 'dq'], kdStop: 3, drawsAllowed: true, legKicks: true, knees: true, legalZones: ['head', 'torso', 'legs'] },
  },
  fouls,
  manualScores: [
    { kind: 'punch', label: 'Award clean punch' },
    { kind: 'kick', label: 'Award clean kick (head / body)' },
    { kind: 'leg_kick', label: 'Award leg kick' },
    { kind: 'knee', label: 'Award knee strike' },
    { kind: 'knockdown', label: 'Knockdown' },
    { kind: 'stop', label: 'Referee stops fight (TKO)' },
  ],
  summary: [
    { title: 'How a bout is scored', items: [
      '10-Point Must system per round: winner 10, loser 9 (10-8 / 10-7 for dominance or knockdowns); 10-10 for an even round (pro).',
      'Scoring priority: knockdowns, then cumulative effective impact, then number of clean scoring strikes, then ring generalship / aggression.',
      'The AI scores clean punches, kicks and knees via the CLIP backend: head or torso above the belt, plus the legs where the variant allows leg kicks. Kicks to the groin or knee joint and (Full Contact) leg kicks are fouls.',
      'Each knockdown costs the downed fighter 1 point; each deduction costs the offender 1 point.',
    ] },
    { title: 'Legal strikes and targets', items: [
      'Pro (Unified): boxing punches, spinning backfists, knees, kicks above the belt, and foot/shin kicks to the legs (not thrusting kicks to the knee joint).',
      'Targets: front and sides of the head; front and sides of the torso above the belt; the leg (pro / Low Kick), but never the knee joint with a thrusting kick.',
      'WAKO Full Contact: strikes above the waist only; no knees, elbows, clinching or throws. WAKO Low Kick adds kicks to the thighs.',
    ] },
    { title: 'Knockdowns', items: [
      'Mandatory 8-count after every knockdown; not up by 10 = KO/TKO. No standing 8-count.',
      '3 knockdowns from head strikes or kicks in a round = TKO. A knocked-down fighter cannot be saved by the bell.',
      'A slip without a preceding fair strike is NOT a knockdown (AI checks for a landed punch in the previous 3 s).',
    ] },
    { title: 'Fouls and penalties', items: [
      'Fouls include: groin strikes, head-butts, elbows, open-glove / non-padded strikes, strikes to the back of the head / spine or throat, striking a downed fighter or after the bell, holding (clinching only to launch a knee, about 5 s), holding a leg without a legal strike, throws / sweeps / takedowns, using the ropes, timidity, intentional delays, thumbing, biting, spitting, abuse, ignoring the referee, thrusting kicks to the knee joint.',
      'Pro: the referee may deduct points or disqualify for intentional, flagrant or repeated fouls (ladder here: caution → −1 → −1 → DQ).',
      'WAKO: warnings, minus points, then disqualification (ladder here: caution → warning → −1 → DQ; edit to your event).',
      'Flagrant fouls and unsportsmanlike conduct can be disqualified immediately. Intentional foul injury that ends the bout = DQ.',
    ] },
    { title: 'How a bout ends', items: ['KO (not up by 10)', 'TKO / RSC (referee stops it; 3 knockdowns in a round)', 'DQ (fouls)', 'Decision on points after the final round (draw allowed in pro)'] },
  ],
  sources: [
    { title: 'ABC Unified Rules of Professional Kickboxing (read)', url: 'https://abcboxing.com/?p=1487' },
    { title: 'WAKO rules (Full Contact ch. 8, Low Kick ch. 9): not read in full', url: 'https://wako.sport/post/new-wako-sport-rules-in-the-web' },
  ],
  verify: [
    'Only the Unified (pro) numbers were read from the primary rulebook. WAKO Full Contact / Low Kick round lengths, penalty ladder, knee legality in Low Kick and knockdown stoppage should be checked against the current WAKO chapters.',
    'Kicks and knees are detected from pose and judged by the CLIP backend like punches. That is new and untested on real footage; use the referee-override buttons whenever it misses or mis-reads one.',
  ],

  // a kick / knee that the backend confirms (or, offline, that the geometry saw land) is scored here
  strikeEvent(ev, m) {
    const v = ev.verdict;
    const kind = ev.type === 'kick_landed' ? ev.strike : v.strike;
    if (m.status === 'rest' || m.status === 'paused') {
      if (!v || (v.strike_check?.value && v.contact.value)) m.penalize(ev.by, 'hit_after_bell', ev);
      return;
    }
    // striking a downed fighter (checked before the "running" test: while he is down the bout is in the 8-count)
    if ((!v || v.strike_check?.value) && this.hitDown(m, ev)) return;
    if (m.status !== 'running') return;
    if (v && !v.strike_check?.value) return void m.say(`Ignored: not a real ${kind} (step / raised leg)`, { kind: 'info', side: ev.by, auto: !!ev.auto });
    if (kind === 'knee' && m.cfg.knees === false && (!v || v.contact.value)) return void m.penalize(ev.by, 'illegal_knee', ev);
    if (!v) { // geometry-only fallback (backend offline)
      if (kind === 'kick' && ev.target === 'leg' && m.cfg.legKicks === false) return void m.penalize(ev.by, 'kick_below_waist', ev);
      const what = kind === 'knee' ? 'knee strike' : ev.target === 'leg' ? 'leg kick' : 'kick';
      return this.awardPunch(m, ev.by, m.other(ev.by), ev.target === 'leg' ? 'leg' : ev.target, true, 'geometry only', ev, what);
    }
    const loc = String(v.contact_location.value).replace(/_/g, ' ');
    const pct = Math.round(v.confidence * 100);
    if (v.foul) m.penalize(ev.by, v.foul, ev);
    else if (v.scored) {
      const what = kind === 'knee' ? 'knee strike' : v.contact_location.value === 'legs' ? 'leg kick' : 'kick';
      this.awardPunch(m, ev.by, m.other(ev.by), loc, true, `${pct}% confidence`, ev, what);
    } else if (v.review) m.say(`REVIEW: ${v.summary}`, { kind: 'info', side: ev.by, big: 'REVIEW', speak: 'Referee review', auto: true, evidence: ev });
    else if (v.contact.value) m.say(`No score: ${v.summary}`, { kind: 'info', side: ev.by, auto: true, evidence: ev });
  },

  handle(ev, m) {
    if (ev.type === 'kick_landed' || (ev.type === 'strike_verdict' && (ev.verdict.strike || 'punch') !== 'punch')) return this.strikeEvent(ev, m);
    if (ev.type === 'score' && KICKS[ev.kind]) {
      if (m.status !== 'running') return;
      if (ev.kind === 'leg_kick' && m.cfg.legKicks === false) return void m.penalize(ev.by, 'kick_below_waist', ev);
      if (ev.kind === 'knee' && m.cfg.knees === false) return void m.penalize(ev.by, 'illegal_knee', ev);
      return this.awardPunch(m, ev.by, m.other(ev.by), ev.kind === 'leg_kick' ? 'leg' : 'head/body', false, 'referee award', ev, KICKS[ev.kind]);
    }
    return boxing.handle.call(this, ev, m);
  },
};
