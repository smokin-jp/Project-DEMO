// Printable match report: one self-contained HTML page (open it, then Print / Save as PDF).
// Pure function: takes the finished (or running) Match plus the saved strike reviews and returns a string.

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x) => `${Math.round((x ?? 0) * 100)}%`;
const yn = (b) => (b === null || b === undefined ? 'n/a' : b ? 'Yes' : 'No');
const word = (s) => String(s ?? '').replace(/_/g, ' ');

function sideSummary(match, id) {
  const s = match.sides[id];
  const id_ = match.rules.id;
  if (id_ === 'judo') return `waza-ari ${s.wazaari} · yuko ${s.yuko} · shido ${s.shido}`;
  return `card ${s.score} pts · clean strikes ${s.punchesTotal} · knockdowns against ${s.kdTotal} · deductions ${s.deductTotal}`;
}

function reviewHtml(r) {
  const v = r.v;
  const kind = r.strike || v.strike || 'punch';
  const side = r.by === 'a' ? 'RED' : 'BLUE';
  const outcome = v.strike_check && !v.strike_check.value ? `not a real ${kind}` : v.foul ? `FOUL: ${word(v.foul)}` : v.scored ? `scoring ${kind}` : v.review ? 'needs referee review' : 'no score';
  const rows = [
    ...(v.strike_check ? [[`Real ${kind}? (not a step)`, yn(v.strike_check.value), v.strike_check.confidence]] : []),
    ['1 Who threw it?', `${v.thrower.detail.name} (${r.hand} ${kind === 'punch' ? 'hand' : kind === 'kick' ? 'leg' : 'knee'})`, v.thrower.confidence],
    ['2 Intended target', v.intended_target.detail.name ?? 'none in frame', null],
    ['3 Contact made?', yn(v.contact.value), v.contact.confidence],
    ['4 Where?', word(v.contact_location.value), v.contact_location.confidence],
    ['5 Legal scoring area?', yn(v.legal_area.value), v.legal_area.confidence],
    [kind === 'punch' ? '6 Legal glove part?' : '6 Legal striking surface?', `${yn(v.legal_glove_part.value)}${v.legal_glove_part.detail.part ? ' · ' + word(v.legal_glove_part.detail.part) : ''}`, v.legal_glove_part.confidence],
    ['7 Genuine strike?', `${yn(v.genuine_punch.value)} · ${v.genuine_punch.detail.classification ?? ''}`, v.genuine_punch.confidence],
    ...(v.strike_check?.detail?.trained_kick ? [['Trained kick detector', `${v.strike_check.detail.trained_kick.positive_frames} of ${v.strike_check.detail.trained_kick.frames_considered} frames show a kick · ${v.strike_check.detail.trained_kick.detected ? 'kick' : 'no kick'}`, v.strike_check.detail.trained_kick.mean]] : []),
    ...(v.punch_type ? [['Punch type (experimental)', v.punch_type.label, v.punch_type.confidence]] : []),
  ];
  const views = (v.views || []).length
    ? `<table class="t small"><tr><th colspan="3">Cameras (each view weighted by how well it sees the strike)</th></tr>${v.views.map((x) => `<tr><td>${esc(x.camera)}</td><td>${x.available ? 'sees both fighters' : 'cannot see the opponent'}</td><td>weight ${pct(x.quality)}${x.p_contact != null ? ` · contact ${pct(x.p_contact)} · ${esc(word(x.location))}` : ''}</td></tr>`).join('')}</table>${v.contact?.detail?.veto_by_good_view ? '<p class="note">A clear miss from a well-placed camera overruled an apparent hit from another angle.</p>' : ''}`
    : '';
  const shots = Object.entries(v.crops || {}).sort(([a], [b]) => a.localeCompare(b)).map(([k, b64]) => `<figure><img alt="${esc(k)}" src="data:image/jpeg;base64,${esc(b64)}"><figcaption>${esc(k.replace('_', ' '))}</figcaption></figure>`).join('');
  return `<section class="review ${v.foul ? 'foul' : v.scored ? 'scored' : v.review ? 'rev' : ''}">
    <h3>#${r.id} · ${side} ${esc(r.hand)} ${esc(kind)} · <span>${esc(outcome)}</span> · ${pct(v.confidence)} <small>${r.round > 1 ? `R${r.round} ` : ''}${esc(r.clock)}</small></h3>
    <p class="sum">${esc(v.summary)}</p>
    ${shots ? `<div class="shots">${shots}</div>` : ''}
    <table class="t">${rows.map(([q, a, c]) => `<tr><td class="q">${esc(q)}</td><td>${esc(a)}</td><td class="c">${c == null ? '' : pct(c)}</td></tr>`).join('')}</table>
    ${views}
    ${r.status ? `<p class="note">Referee decision on this review: <b>${esc(r.status)}</b></p>` : ''}
  </section>`;
}

export function buildReportHtml({ match, reviews = [], generatedAt = new Date() }) {
  const rules = match.rules;
  const ended = match.status === 'ended';
  const result = ended ? (match.winner ? `${match.sides[match.winner].name} wins by ${match.method}` : `No winner: ${match.method}`) : 'Bout not finished';
  const cards = match.roundCards || [];
  const scorecard = cards.length
    ? `<h2>Scorecard (10-point must)</h2><table class="t"><tr><th>Round</th><th>RED</th><th>BLUE</th></tr>${cards.map((r) => {
      const c = (x) => `<b>${x.pts}</b> <small>${x.strikes} strikes${x.kd ? ` · KD ${x.kd}` : ''}${x.ded ? ` · −${x.ded}` : ''}</small>`;
      return `<tr><td>${r.round}</td><td>${c(r.a)}</td><td>${c(r.b)}</td></tr>`;
    }).join('')}<tr class="tot"><td>Total</td><td><b>${match.sides.a.score}</b></td><td><b>${match.sides.b.score}</b></td></tr></table>`
    : '';
  const penalties = ['a', 'b'].map((id) => {
    const p = match.sides[id].penalties;
    return `<div><h3 class="${id}">${match.sides[id].name}</h3>${p.length ? `<ul>${p.map((x) => `<li>${esc(x.label)} <b>→ ${esc(x.action)}</b></li>`).join('')}</ul>` : '<p class="dim">no penalties</p>'}</div>`;
  }).join('');
  const timeline = [...match.log].reverse().map((l) => `<tr><td class="tm">${l.round > 1 ? `R${l.round} ` : ''}${esc(l.clock)}</td><td>${l.auto ? '<span class="ai">AI</span>' : l.kind === 'info' ? '' : '<span class="ref">REF</span>'}</td><td class="${esc(l.kind)}">${esc(l.text)}</td></tr>`).join('');
  const revs = [...reviews].reverse().map(reviewHtml).join('');
  const when = generatedAt.toLocaleString();

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>AI Referee: match report</title>
<style>
  body { font: 14px/1.45 system-ui, 'Segoe UI', Arial, sans-serif; color: #17202a; max-width: 900px; margin: 24px auto; padding: 0 18px; }
  h1 { margin: 0 0 2px; font-size: 24px; } h2 { margin: 26px 0 8px; font-size: 17px; border-bottom: 2px solid #17202a; padding-bottom: 3px; } h3 { margin: 0 0 4px; font-size: 14px; }
  .meta, .dim, small { color: #6b7785; } .a { color: #d62839; } .b { color: #1e5fd8; }
  .result { font-size: 20px; font-weight: 800; background: #fff6d6; border: 2px solid #e0a800; border-radius: 10px; padding: 10px 14px; margin: 14px 0; }
  .sides { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; } .side { border: 1px solid #cfd6dd; border-left-width: 6px; border-radius: 8px; padding: 8px 12px; }
  .side.a { border-left-color: #d62839; } .side.b { border-left-color: #1e5fd8; } .side b { font-size: 18px; }
  table.t { width: 100%; border-collapse: collapse; margin: 6px 0; } .t td, .t th { border: 1px solid #dde2e7; padding: 4px 8px; text-align: left; vertical-align: top; } .t th { background: #f1f4f7; }
  .t .q { color: #6b7785; width: 32%; } .t .c { text-align: right; width: 70px; } .t tr.tot td { background: #f1f4f7; font-weight: 700; } .small { font-size: 12px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; } ul { margin: 4px 0; padding-left: 18px; }
  .tm { color: #6b7785; font-family: ui-monospace, Consolas, monospace; white-space: nowrap; width: 70px; } .ai { background: #dbe8ff; color: #1e4fb5; border-radius: 4px; padding: 0 5px; font-size: 11px; } .ref { background: #e7e7e7; border-radius: 4px; padding: 0 5px; font-size: 11px; }
  td.score { color: #11824f; } td.foul { color: #c25a00; } td.result { color: #8a6500; font-weight: 700; }
  .review { border: 1px solid #cfd6dd; border-left: 6px solid #cfd6dd; border-radius: 8px; padding: 10px 14px; margin: 12px 0; break-inside: avoid; } .review.scored { border-left-color: #11824f; } .review.foul { border-left-color: #c25a00; } .review.rev { border-left-color: #e0a800; }
  .review h3 span { font-weight: 800; } .sum { margin: 2px 0 8px; } .note { background: #f1f4f7; border-radius: 6px; padding: 5px 9px; font-size: 12px; }
  .shots { display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0; } .shots figure { margin: 0; text-align: center; font-size: 11px; color: #6b7785; } .shots img { width: 120px; height: 120px; object-fit: cover; border-radius: 6px; border: 1px solid #cfd6dd; display: block; }
  .foot { margin-top: 28px; padding-top: 10px; border-top: 1px solid #cfd6dd; font-size: 12px; color: #6b7785; }
  .noprint button { font: inherit; padding: 8px 16px; border-radius: 8px; border: 1px solid #17202a; background: #17202a; color: #fff; cursor: pointer; }
  @media print { .noprint { display: none; } body { margin: 0; max-width: none; } }
</style></head><body>
<div class="noprint" style="float:right"><button onclick="window.print()">🖨 Print / save as PDF</button></div>
<h1>AI Referee: match report</h1>
<div class="meta">${esc(rules.icon)} ${esc(rules.name)} · ${esc(match.cfg.label)} · generated ${esc(when)}</div>
<div class="result">${esc(result)}</div>
<div class="sides">${['a', 'b'].map((id) => `<div class="side ${id}"><div class="${id}"><b>${match.sides[id].name}</b></div>${esc(sideSummary(match, id))}</div>`).join('')}</div>
${scorecard}
<h2>Penalties</h2><div class="grid2">${penalties}</div>
<h2>Timeline</h2><table class="t small">${timeline || '<tr><td class="dim">no calls yet</td></tr>'}</table>
<h2>Judged strikes (${reviews.length})</h2>${revs || '<p class="dim">No strikes were judged by the CLIP backend in this bout.</p>'}
<div class="foot"><b>How to read this:</b> calls tagged <span class="ai">AI</span> were made by the system (pose tracking in the browser, OpenAI CLIP and pose geometry on the backend); <span class="ref">REF</span> are manual referee actions. The system is an assistant: the human referee's decision is final, and any automatic penalty can be withdrawn with Undo. Detection has been tested on synthetic data; confidence figures are uncalibrated.</div>
</body></html>`;
}
