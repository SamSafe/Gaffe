export function newsPanel(){return '<h1>Team news</h1><section id="news-content" aria-live="polite">Loading fixture reports…</section>';}

export async function bindTeamNews({$, api, players, esc, points, rosterData, toast}) {
    const panel = $('#news-content');
    if (!panel) return;
    try {
        const result = await api('team-news');
        if (!panel.isConnected) return;
        const playerName = id => esc(players[id]?.name || id);
        const names = ids => ids.length ? ids.map(playerName).join(', ') : 'None';
        const option = (id, label) => `<option value="${Number(id)}">${esc(label)}</option>`;
        const reportList = result.reports.map(r => `<p><strong>${playerName(r.player_id)}</strong> · ${r.cap} min cap${r.replacement_id ? ` · replacement ${playerName(r.replacement_id)}` : ''}<br>${esc(r.quote)} <a href="${esc(r.source)}" target="_blank" rel="noopener noreferrer">Source</a></p>`).join('');
        const scores = result.scores.map(s => `<tr><td>${esc(s.role)}</td><td>${esc(s.arm)}</td><td>${s.n}</td><td>${s.brier.toFixed(3)}</td><td>${points(s.minutes_mae)}</td><td>${points(s.points_mae)}</td></tr>`).join('');
        const inbox = result.inbox.map((r,i) => `<p>${esc(r.quote)} <a href="${esc(r.source)}" target="_blank" rel="noopener noreferrer">BBC Sport</a> <button type="button" class="button secondary" data-story="${i}">Review for ${playerName(r.player_id)}</button></p>`).join('');
        const history = result.history.map(h => {
            const compare = ['baseline','news','reranked'].map(arm => {
                const d = h.decisions[arm];
                return d ? `<details><summary>${({baseline:'Baseline',news:'News with original shortlist',reranked:'News with refreshed shortlist'})[arm]} squad</summary><p>Captain · ${playerName(d.captain)}<br>Starting XI · ${names(d.starting_xi)}<br>Transfers in · ${names(d.transferred_in)}<br>Transfers out · ${names(d.transferred_out)}</p></details>` : '';
            }).join('');
            const pool=h.shortlist;
            const fixed=h.decisions.news, refreshed=h.decisions.reranked;
            const changed=fixed && refreshed && ['transferred_in','transferred_out'].some(k=>JSON.stringify(fixed[k])!==JSON.stringify(refreshed[k]));
            const signal=pool?.entered ? `<p>Entered shortlist: ${pool.entered.map(id=>playerName(id)+(h.entrant_points?.[id]!=null?' ('+h.entrant_points[id]+' GW pts)':'')).join(', ')||'None'}<br>Left shortlist: ${names(pool.left)}${fixed&&refreshed?'<br>Transfer recommendation '+(changed?'changed':'unchanged'):''}</p>` : '';
            return `<section><h3>GW${h.gw}</h3>${signal}${h.rows.map(r => `<p><strong>${playerName(r.player_id)}</strong> (${esc(r.role)}) · ${points(r.expected_minutes)} → ${points(r.expected_minutes_news)} minutes · ${points(r.e_xpts)} → ${points(r.e_xpts_news)} xPts</p>`).join('')}${compare}${h.skipped.map(s=>`<p>${esc(s)}</p>`).join('')}</section>`;
        }).join('');
        const field = (label, input) => `<div class="field"><label>${label}${input}</label></div>`;
        const form = `<form id="news-form">
            ${field('Fixture', `<select name="fixture" required>${result.games.map(g=>option(g.id,g.name)).join('')}</select>`)}
            ${field('Reported player','<select name="player" required></select>')}
            ${field('Likely replacement','<select name="replacement"></select>')}
            ${field('Source URL','<input type="url" name="source" required>')}
            ${field('Published at (your local time)','<input type="datetime-local" name="published" required>')}
            ${field('Exact quote','<textarea name="quote" required minlength="3" maxlength="1200"></textarea>')}
            ${field('Minutes cap (0 if ruled out)','<input type="number" name="cap" min="0" max="89" placeholder="Leave blank to extract an explicit limit">')}
            ${field('Withdraw earlier report for this player and fixture','<input type="checkbox" name="withdraw">')}
            <button class="button">Save report</button></form>`;
        panel.innerHTML = `<div class="settings-grid"><section class="card"><div class="card-head"><h2>Record team news</h2></div><div class="card-body">${result.games.length ? form : 'Refresh player and fixture data to record a report.'}</div></section>
            <div class="stack"><section class="card"><div class="card-head"><h2>Upcoming reports</h2></div><div class="card-body">${reportList || 'No eligible reports for this deadline.'}<p>${esc(result.feed_status)}</p>${inbox}<p>Generate a forecast to compare these reports with the regular model.</p></div></section>
            <section class="card"><div class="card-head"><h2>Forecast comparisons</h2></div><div class="card-body">${history || 'No paired forecasts yet.'}</div></section>
            <section class="card"><div class="card-head"><h2>Finalized results</h2></div><div class="card-body">${scores ? `<div class="table-wrap"><table><thead><tr><th>Player group</th><th>Forecast</th><th>Fixtures</th><th>Minutes Brier</th><th>Minutes MAE</th><th>Points MAE</th></tr></thead><tbody>${scores}</tbody></table></div>` : 'Results appear after FPL finalizes a gameweek and you refresh.'}</div></section></div></div>`;
        const entry = $('#news-form');
        if (!entry) return;
        const fields = entry.elements;
        const roster = rosterData.slice().sort((a,b)=>a.name.localeCompare(b.name));
        const selectReplacement = () => {
            const p = players[fields.player.value];
            fields.replacement.innerHTML = '<option value="">Unspecified</option>' + roster.filter(x=>p && x.club_id===p.club_id && x.id!==p.id).map(x=>option(x.id,x.name)).join('');
        };
        const selectFixture = () => {
            const clubs = result.games.find(g=>g.id===Number(fields.fixture.value)).clubs;
            fields.player.innerHTML = roster.filter(p=>clubs.includes(p.club_id)).map(p=>option(p.id,`${p.name} · ${p.club}`)).join('');
            selectReplacement();
        };
        selectFixture();
        panel.querySelectorAll('[data-story]').forEach(button => {
            button.onclick = () => {
                const r = result.inbox[Number(button.dataset.story)];
                fields.fixture.value = String(r.fixture_id);
                selectFixture();
                fields.player.value = String(r.player_id);
                selectReplacement();
                fields.source.value = r.source;
                fields.quote.value = r.quote;
                const t = new Date(r.published_at);
                fields.published.value = new Date(t.getTime() - t.getTimezoneOffset()*60000).toISOString().slice(0,16);
                fields.cap.value = '';
                fields.withdraw.checked = false;
                fields.cap.focus();
            };
        });
        fields.fixture.onchange = selectFixture;
        fields.player.onchange = selectReplacement;
        entry.onsubmit = async e => {
            e.preventDefault();
            const button = entry.querySelector('button');
            button.disabled = true;
            try {
                await api('team-news','POST',{fixture_id:Number(fields.fixture.value),player_id:Number(fields.player.value),replacement_id:Number(fields.replacement.value)||null,source:fields.source.value,published_at:new Date(fields.published.value).toISOString(),quote:fields.quote.value,cap:fields.cap.value===''?null:Number(fields.cap.value),withdraw:fields.withdraw.checked});
                toast('Team-news report saved.');
                await bindTeamNews({$, api, players, esc, points, rosterData, toast});
            } catch (err) { toast(err.message); button.disabled = false; }
        };
    } catch (err) { panel.textContent = err.message; }
}
