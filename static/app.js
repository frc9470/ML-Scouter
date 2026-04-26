let points = [];
let imgNativeWidth = 0;
let imgNativeHeight = 0;

document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('roi-canvas');
    const ctx = canvas.getContext('2d');
    const startButton = document.getElementById('btn-start');
    const sourceStatus = document.getElementById('source-status');
    const tbaStatus = document.getElementById('tba-status');
    const frameStatus = document.getElementById('frame-status');
    const frameSecondsInput = document.getElementById('roi-frame-seconds');
    const processSecondsInput = document.getElementById('process-seconds');
    const frameError = document.getElementById('first-frame-error');
    const currentYear = new Date().getFullYear();
    document.getElementById('tba-year').value = currentYear;

    // ──── Combobox helper ────
    function createCombobox({ inputId, listboxId, spinnerId, onSearch, onSelect, debounceMs = 300 }) {
        const input = document.getElementById(inputId);
        const listbox = document.getElementById(listboxId);
        const spinner = document.getElementById(spinnerId);
        let items = [], focusIdx = -1, timer = null, selectedValue = null;

        function open() { if (listbox.children.length) listbox.classList.add('open'); }
        function close() { listbox.classList.remove('open'); focusIdx = -1; }
        function setSpinner(on) { spinner.classList.toggle('active', on); }

        function render(newItems, emptyMsg) {
            items = newItems;
            listbox.innerHTML = '';
            focusIdx = -1;
            if (!items.length) {
                if (emptyMsg) listbox.innerHTML = `<li class="combobox-empty">${emptyMsg}</li>`;
                open(); return;
            }
            items.forEach((item, i) => {
                const li = document.createElement('li');
                li.className = 'combobox-option';
                li.setAttribute('role', 'option');
                li.innerHTML = item.html;
                li.addEventListener('mousedown', e => { e.preventDefault(); pick(i); });
                listbox.appendChild(li);
            });
            open();
        }

        function pick(i) {
            const item = items[i];
            if (!item) return;
            selectedValue = item.value;
            input.value = item.text;
            close();
            if (onSelect) onSelect(item);
        }

        function scrollTo(i) {
            const el = listbox.children[i];
            if (el) el.scrollIntoView({ block: 'nearest' });
        }

        function setFocus(i) {
            Array.from(listbox.children).forEach(c => c.classList.remove('focused'));
            focusIdx = i;
            if (listbox.children[i]) { listbox.children[i].classList.add('focused'); scrollTo(i); }
        }

        input.addEventListener('input', () => {
            clearTimeout(timer);
            timer = setTimeout(() => { setSpinner(true); onSearch(input.value.trim(), (items, msg) => { setSpinner(false); render(items, msg); }); }, debounceMs);
        });

        input.addEventListener('focus', () => { if (listbox.children.length) open(); });
        input.addEventListener('blur', () => setTimeout(close, 150));

        input.addEventListener('keydown', e => {
            const len = items.length;
            if (e.key === 'ArrowDown') { e.preventDefault(); setFocus(focusIdx < len - 1 ? focusIdx + 1 : 0); }
            else if (e.key === 'ArrowUp') { e.preventDefault(); setFocus(focusIdx > 0 ? focusIdx - 1 : len - 1); }
            else if (e.key === 'Enter') { e.preventDefault(); if (focusIdx >= 0) pick(focusIdx); }
            else if (e.key === 'Escape') { close(); input.blur(); }
        });

        return {
            render, close, setSpinner,
            getSelected: () => selectedValue,
            setPlaceholder: t => { input.placeholder = t; },
            enable: () => { input.disabled = false; },
            disable: () => { input.disabled = true; },
            clear: () => { input.value = ''; selectedValue = null; listbox.innerHTML = ''; close(); }
        };
    }

    // ──── TBA Event combobox ────
    let allEvents = [];
    const eventCombo = createCombobox({
        inputId: 'tba-event-query', listboxId: 'event-listbox', spinnerId: 'event-spinner',
        debounceMs: 400,
        onSearch: (query, done) => {
            const year = document.getElementById('tba-year').value.trim();
            fetch(`/api/tba/events?year=${encodeURIComponent(year)}&q=${encodeURIComponent(query)}`)
                .then(r => r.json().then(d => ({ ok: r.ok, d })))
                .then(({ ok, d }) => {
                    if (!ok) { done([], d.error || 'Error searching events'); return; }
                    allEvents = d.events;
                    done(d.events.map(ev => {
                        const place = [ev.city, ev.state_prov, ev.country].filter(Boolean).join(', ');
                        return {
                            value: ev.key,
                            text: ev.name,
                            html: `<span class="combobox-option-label">${esc(ev.name)}</span>` +
                                  (place ? `<span class="combobox-option-meta">${esc(ev.key)} · ${esc(place)}</span>` : `<span class="combobox-option-meta">${esc(ev.key)}</span>`)
                        };
                    }), d.events.length ? null : 'No matching events found');
                })
                .catch(() => done([], 'Could not reach TBA API'));
        },
        onSelect: (item) => {
            tbaStatus.textContent = `Loading matches for ${item.text}…`;
            tbaStatus.className = 'source-status active';
            matchCombo.clear();
            matchCombo.disable();
            matchCombo.setSpinner(true);
            fetch(`/api/tba/event/${encodeURIComponent(item.value)}/matches`)
                .then(r => r.json().then(d => ({ ok: r.ok, d })))
                .then(({ ok, d }) => {
                    matchCombo.setSpinner(false);
                    if (!ok) { tbaStatus.textContent = d.error || 'Error loading matches'; tbaStatus.className = 'source-status error'; return; }
                    allMatches = d.matches;
                    matchCombo.enable();
                    matchCombo.setPlaceholder(`Search ${d.matches.length} matches…`);
                    matchCombo.render(matchItemsFromData(d.matches), d.matches.length ? null : 'No match videos found');
                    tbaStatus.textContent = `${d.matches.length} match videos found.`;
                    tbaStatus.className = d.matches.length ? 'source-status success' : 'source-status error';
                })
                .catch(() => { matchCombo.setSpinner(false); tbaStatus.textContent = 'Could not load matches.'; tbaStatus.className = 'source-status error'; });
        }
    });

    // ──── TBA Match combobox ────
    let allMatches = [];

    function matchItemsFromData(matches) {
        return matches.map(m => {
            const redBadges = m.red.map(t => `<span class="combobox-badge red">${esc(t)}</span>`).join('');
            const blueBadges = m.blue.map(t => `<span class="combobox-badge blue">${esc(t)}</span>`).join('');
            return {
                value: m.youtube_url,
                text: m.label,
                html: `<span class="combobox-option-label">${esc(m.label)}</span>` +
                      (redBadges || blueBadges ? `<div class="combobox-teams">${redBadges}${blueBadges}</div>` : '')
            };
        });
    }

    const matchCombo = createCombobox({
        inputId: 'tba-match-query', listboxId: 'match-listbox', spinnerId: 'match-spinner',
        debounceMs: 0,
        onSearch: (query, done) => {
            const q = query.toLowerCase();
            const filtered = q ? allMatches.filter(m =>
                m.label.toLowerCase().includes(q) ||
                m.red.some(t => t.includes(q)) ||
                m.blue.some(t => t.includes(q))
            ) : allMatches;
            done(matchItemsFromData(filtered), filtered.length ? null : 'No matches found');
        },
        onSelect: (item) => {
            if (!item.value) { tbaStatus.textContent = 'No video URL for this match.'; tbaStatus.className = 'source-status error'; return; }
            document.getElementById('youtube-url').value = item.value;
            loadYoutubeUrl(item.value);
        }
    });

    function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

    // ──── TBA config ────
    function refreshTbaConfigStatus() {
        fetch('/api/tba/config').then(r => r.json()).then(d => {
            if (d.configured) { tbaStatus.textContent = 'TBA key configured.'; tbaStatus.className = 'source-status success'; }
            else { tbaStatus.textContent = 'Enter a TBA API key to use the event picker.'; tbaStatus.className = 'source-status active'; }
        }).catch(() => { tbaStatus.textContent = 'Could not check TBA key status.'; tbaStatus.className = 'source-status error'; });
    }
    refreshTbaConfigStatus();

    // ──── ROI helpers ────
    function clearRoi() { points = []; drawPoints(); startButton.disabled = true; }
    function setFrameError(message) { frameError.textContent = message; frameError.classList.add('active'); }
    function clearFrameError() { frameError.textContent = ''; frameError.classList.remove('active'); }

    function setFirstFrame(image) {
        clearFrameError();
        const img = document.getElementById('first-frame');
        img.src = image;
        img.onload = () => { imgNativeWidth = img.naturalWidth; imgNativeHeight = img.naturalHeight; resizeCanvas(); };
    }

    function setFrameSeconds(seconds) { frameSecondsInput.value = Math.max(0, Number(seconds) || 0).toFixed(1); }

    function loadFirstFrame() {
        fetch('/api/first_frame').then(r => r.json().then(d => ({ ok: r.ok, d }))).then(({ ok, d }) => {
            if (!ok) { setFrameError(d.error || 'Could not load the first video frame.'); return; }
            if (d.image) { setFrameSeconds(d.seconds || 0); setFirstFrame(d.image); }
        }).catch(() => setFrameError('Could not connect to the video frame endpoint.'));
    }
    loadFirstFrame();

    function resizeCanvas() {
        const img = document.getElementById('first-frame');
        canvas.width = img.clientWidth; canvas.height = img.clientHeight; drawPoints();
    }
    window.addEventListener('resize', resizeCanvas);

    canvas.addEventListener('mousedown', (e) => {
        if (!imgNativeWidth || !imgNativeHeight) return;
        const rect = canvas.getBoundingClientRect();
        const scaleX = imgNativeWidth / canvas.width, scaleY = imgNativeHeight / canvas.height;
        const x = (e.clientX - rect.left) * scaleX, y = (e.clientY - rect.top) * scaleY;
        points.push([Math.round(x), Math.round(y)]);
        drawPoints();
        if (points.length >= 3) startButton.disabled = false;
    });

    function drawPoints() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!imgNativeWidth || !imgNativeHeight) return;
        const scaleX = canvas.width / imgNativeWidth, scaleY = canvas.height / imgNativeHeight;
        if (points.length > 0) {
            ctx.beginPath(); ctx.strokeStyle = '#00ffff'; ctx.lineWidth = 2;
            points.forEach((pt, i) => { const x = pt[0] * scaleX, y = pt[1] * scaleY; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); ctx.fillStyle = '#ff0000'; });
            if (points.length > 2) ctx.lineTo(points[0][0] * scaleX, points[0][1] * scaleY);
            ctx.stroke();
            points.forEach(pt => { const x = pt[0] * scaleX, y = pt[1] * scaleY; ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); });
        }
    }

    document.getElementById('btn-clear').addEventListener('click', () => clearRoi());

    // ──── Frame navigation ────
    function loadFrameAtSeconds(seconds) {
        frameStatus.textContent = 'Loading ROI frame…'; frameStatus.className = 'source-status active';
        document.getElementById('btn-load-frame').disabled = true; startButton.disabled = true;
        fetch('/api/frame_at', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ seconds }) })
            .then(r => r.json().then(d => ({ ok: r.ok, d })))
            .then(({ ok, d }) => {
                if (!ok) { frameStatus.textContent = d.error || 'Could not load ROI frame.'; frameStatus.className = 'source-status error'; return; }
                setFrameSeconds(d.seconds); clearRoi(); setFirstFrame(d.image);
                frameStatus.textContent = `ROI frame loaded at ${Number(d.seconds).toFixed(1)}s.`; frameStatus.className = 'source-status success';
            })
            .catch(() => { frameStatus.textContent = 'Could not connect to the ROI frame endpoint.'; frameStatus.className = 'source-status error'; })
            .finally(() => { document.getElementById('btn-load-frame').disabled = false; });
    }

    document.getElementById('btn-load-frame').addEventListener('click', () => loadFrameAtSeconds(Number(frameSecondsInput.value) || 0));
    document.getElementById('btn-frame-back').addEventListener('click', () => loadFrameAtSeconds(Math.max(0, (Number(frameSecondsInput.value) || 0) - 5)));
    document.getElementById('btn-frame-forward').addEventListener('click', () => loadFrameAtSeconds((Number(frameSecondsInput.value) || 0) + 5));

    // ──── YouTube loader ────
    function loadYoutubeUrl(url) {
        if (!url) { sourceStatus.textContent = 'Paste a YouTube URL first.'; sourceStatus.className = 'source-status error'; return; }
        sourceStatus.textContent = 'Downloading video…'; sourceStatus.className = 'source-status active';
        document.getElementById('btn-load-youtube').disabled = true; startButton.disabled = true;
        fetch('/api/set_video_source', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ youtube_url: url }) })
            .then(r => r.json().then(d => ({ ok: r.ok, d })))
            .then(({ ok, d }) => {
                if (!ok) { sourceStatus.textContent = d.error || 'Could not load YouTube video.'; sourceStatus.className = 'source-status error'; return; }
                sourceStatus.textContent = 'Video loaded. Pick an ROI frame or select the ROI on this frame.'; sourceStatus.className = 'source-status success';
                frameStatus.textContent = 'Title card? Jump forward a few seconds, then load the ROI frame.'; frameStatus.className = 'source-status active';
                setFrameSeconds(d.seconds || 0); clearRoi(); setFirstFrame(d.image);
            })
            .catch(() => { sourceStatus.textContent = 'Could not connect to the video source endpoint.'; sourceStatus.className = 'source-status error'; })
            .finally(() => { document.getElementById('btn-load-youtube').disabled = false; });
    }

    document.getElementById('btn-load-youtube').addEventListener('click', () => loadYoutubeUrl(document.getElementById('youtube-url').value.trim()));

    // ──── TBA key save ────
    document.getElementById('btn-save-tba-key').addEventListener('click', () => {
        const keyInput = document.getElementById('tba-auth-key');
        const key = keyInput.value.trim();
        if (!key) { tbaStatus.textContent = 'Paste a TBA API key first.'; tbaStatus.className = 'source-status error'; return; }
        tbaStatus.textContent = 'Saving TBA key…'; tbaStatus.className = 'source-status active';
        document.getElementById('btn-save-tba-key').disabled = true;
        fetch('/api/tba/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ auth_key: key }) })
            .then(r => r.json().then(d => ({ ok: r.ok, d })))
            .then(({ ok, d }) => {
                if (!ok) { tbaStatus.textContent = d.error || 'Could not save TBA key.'; tbaStatus.className = 'source-status error'; return; }
                keyInput.value = ''; tbaStatus.textContent = 'TBA key saved.'; tbaStatus.className = 'source-status success';
            })
            .catch(() => { tbaStatus.textContent = 'Could not connect to the TBA key endpoint.'; tbaStatus.className = 'source-status error'; })
            .finally(() => { document.getElementById('btn-save-tba-key').disabled = false; });
    });

    // 2. Start Processing
    startButton.addEventListener('click', () => {
        const processSeconds = Math.max(0, Number(processSecondsInput.value) || 0);
        fetch('/api/set_roi', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points, process_seconds: processSeconds }) })
            .then(r => r.json()).then(d => {
                if (d.success) { document.getElementById('phase-roi').classList.remove('active'); document.getElementById('phase-processing').classList.add('active'); pollStatus(); }
            });
    });

    // 3. Poll Backend
    function pollStatus() {
        const interval = setInterval(() => {
            fetch('/api/status').then(r => r.json()).then(d => {
                const percent = d.total_frames > 0 ? Math.min(100, Math.round((d.progress / d.total_frames) * 100)) : 0;
                document.getElementById('progress-bar').style.width = percent + '%';
                document.getElementById('progress-text').innerText = `${percent}% (${d.progress}/${d.total_frames})`;
                if (d.current_fuel_count !== undefined) document.getElementById('live-fuel-count').innerText = d.current_fuel_count;
                if (d.preview_available) {
                    const lp = document.getElementById('live-preview');
                    lp.src = '/api/live_frame?t=' + Date.now(); lp.classList.add('active');
                    document.getElementById('live-preview-placeholder').classList.add('hidden');
                }
                if (d.is_finished) {
                    clearInterval(interval);
                    document.getElementById('phase-processing').classList.remove('active');
                    document.getElementById('phase-playback').classList.add('active');
                    document.getElementById('video-source').src = '/static/output.mp4?t=' + Date.now();
                    document.getElementById('video-player').load();
                }
            });
        }, 1000);
    }

    // 4. Continue to attribution
    document.getElementById('btn-continue-attribution').addEventListener('click', () => showResults());

    // 5. Show Robot Attribution Screen
    function showResults() {
        document.getElementById('phase-playback').classList.remove('active');
        document.getElementById('phase-results').classList.add('active');
        fetch('/api/results').then(r => r.json()).then(d => {
            const grid = document.getElementById('robot-grid');
            Object.keys(d.crops).forEach(r_id => {
                const div = document.createElement('div'); div.className = 'robot-card';
                div.innerHTML = `<img src="data:image/jpeg;base64,${d.crops[r_id]}" /><p style="margin:0 0 10px 0; color:#94a3b8">Tracker ID: ${r_id}</p><p style="margin:0 0 10px 0; color:#ec4899; font-weight:bold">${d.scores[r_id] || 0} Points Scored</p><input type="number" class="input-field" data-id="${r_id}" placeholder="Enter Team #" />`;
                grid.appendChild(div);
            });
        });
    }

    // 6. Submit Scores
    document.getElementById('btn-submit-scores').addEventListener('click', () => {
        const inputs = document.querySelectorAll('.input-field');
        const mapping = {};
        inputs.forEach(input => { if (input.value) mapping[input.dataset.id] = input.value; });
        fetch('/api/submit_attribution', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mapping }) })
            .then(r => r.json()).then(d => {
                document.getElementById('phase-results').classList.remove('active');
                document.getElementById('phase-final').classList.add('active');
                const container = document.getElementById('final-scores-container');
                let html = '';
                Object.keys(d.final_scores).sort((a, b) => d.final_scores[b] - d.final_scores[a]).forEach(team => {
                    html += `<div class="score-row"><span>Team ${team}</span><span style="color:var(--accent); font-weight:bold">${d.final_scores[team]} FUEL</span></div>`;
                });
                html += `<div class="score-row" style="opacity:0.7"><span>Unattributed / Manual</span><span>${d.unattributed}</span></div>`;
                container.innerHTML = html;
            });
    });
});
