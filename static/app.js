let points = [];
let imgNativeWidth = 0;
let imgNativeHeight = 0;
let perspectiveConfig = null;
let baselinePerspectiveConfig = null;
let activeCalibrationView = null;
const calibrationColors = {
    full: '#e0b450',
    left: '#50befa',
    right: '#fa8c50'
};

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
    const calibrationStatus = document.getElementById('calibration-status');
    const analysisRunPanel = document.getElementById('analysis-run-panel');
    const analysisRunSelect = document.getElementById('analysis-run-select');
    const analysisRunSummary = document.getElementById('analysis-run-summary');
    const analysisRunStatus = document.getElementById('analysis-run-status');
    const currentYear = new Date().getFullYear();
    document.getElementById('tba-year').value = currentYear;
    let analysisRuns = [];

    function cloneJson(value) {
        return JSON.parse(JSON.stringify(value));
    }

    function loadPerspectiveConfig() {
        fetch('/api/perspective_config')
            .then(r => r.json())
            .then(config => {
                perspectiveConfig = config;
                baselinePerspectiveConfig = cloneJson(config);
                renderCalibrationControls();
                drawPoints();
            })
            .catch(() => {
                calibrationStatus.textContent = 'Could not load perspective calibration.';
                calibrationStatus.className = 'source-status error';
            });
    }
    loadPerspectiveConfig();

    // ──── Phase helpers ────
    function showPhase(id) {
        document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));
        document.getElementById(id).classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function selectedTrackerMode() {
        const checked = document.querySelector('input[name="tracker-mode"]:checked');
        return checked ? checked.value : 'botsort';
    }

    function withCacheBuster(url) {
        return `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`;
    }

    function loadPlaybackVideos(run) {
        if (!run || !run.video_urls) return;
        document.getElementById('video-source').src = withCacheBuster(run.video_urls.annotated);
        document.getElementById('video-player').load();
        document.getElementById('yolo-debug-source').src = withCacheBuster(run.video_urls.yolo_debug);
        document.getElementById('yolo-debug-player').load();
        document.getElementById('projection-source').src = withCacheBuster(run.video_urls.projection);
        document.getElementById('projection-player').load();
    }

    function renderAnalysisRuns(runs, activeKey) {
        analysisRuns = Array.isArray(runs) ? runs : [];
        if (!analysisRuns.length) {
            analysisRunPanel.style.display = 'none';
            analysisRunSummary.innerHTML = '';
            analysisRunSelect.innerHTML = '';
            return;
        }

        analysisRunPanel.style.display = '';
        analysisRunSelect.innerHTML = analysisRuns
            .map(run => `<option value="${run.key}" ${run.key === activeKey ? 'selected' : ''}>${esc(run.label)}</option>`)
            .join('');
        analysisRunSummary.innerHTML = analysisRuns.map(run => {
            const activeClass = run.key === activeKey ? ' active' : '';
            return `<div class="comparison-card${activeClass}">
                <h4>${esc(run.label)}</h4>
                <p>${esc(run.tracker_config || '')}</p>
                <div class="comparison-metric"><span>Scored Fuel</span><strong>${run.total_scored}</strong></div>
                <div class="comparison-metric"><span>Unique Fuel IDs</span><strong>${run.unique_fuel_tracks}</strong></div>
                <div class="comparison-metric"><span>Fuel Detections</span><strong>${run.fuel_detections}</strong></div>
                <div class="comparison-metric"><span>Robot Paths</span><strong>${run.robot_path_count}</strong></div>
                <div class="comparison-metric"><span>Max Live Fuel</span><strong>${run.max_live_fuel_count}</strong></div>
                <div class="comparison-metric"><span>Runtime</span><strong>${run.runtime_seconds}s</strong></div>
            </div>`;
        }).join('');

        const activeRun = analysisRuns.find(run => run.key === activeKey) || analysisRuns[0];
        analysisRunStatus.className = 'source-status success';
        analysisRunStatus.textContent = `${activeRun.label} is active for playback and attribution.`;
        loadPlaybackVideos(activeRun);
    }

    function setActiveAnalysisRun(runKey) {
        return fetch('/api/select_analysis_run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ run_key: runKey })
        })
            .then(r => r.json().then(d => ({ ok: r.ok, d })))
            .then(({ ok, d }) => {
                if (!ok) throw new Error(d.error || 'Could not switch analysis run');
                renderAnalysisRuns(d.analysis_runs, d.active_analysis_run);
            });
    }

    function advanceToRoi(imageData, seconds) {
        setFrameSeconds(seconds || 0);
        clearRoi();
        setFirstFrame(imageData);
        showPhase('phase-roi');
    }

    // Check for pre-loaded sample video on startup
    let sampleFrameData = null;
    let sampleSeconds = 0;
    fetch('/api/first_frame')
        .then(r => r.json().then(d => ({ ok: r.ok, d })))
        .then(({ ok, d }) => {
            if (ok && d.image) {
                sampleFrameData = d.image;
                sampleSeconds = d.seconds || 0;
                const banner = document.getElementById('sample-banner');
                banner.style.display = '';
                const name = (d.video_path || '').split('/').pop() || 'sample video';
                document.getElementById('sample-banner-name').textContent = name;
            }
        })
        .catch(() => {});

    document.getElementById('btn-use-sample').addEventListener('click', () => {
        if (sampleFrameData) advanceToRoi(sampleFrameData, sampleSeconds);
    });

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
                            match: m,
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
            setManualTeamInputs(item.match ? item.match.red : [], item.match ? item.match.blue : []);
            loadYoutubeUrl(item.value, item.match);
        }
    });

    function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

    function parseTeamList(value) {
        return (value || '')
            .split(/[\s,;]+/)
            .map(item => item.replace(/\D/g, ''))
            .filter(Boolean);
    }

    function setManualTeamInputs(red, blue) {
        document.getElementById('manual-red-teams').value = (red || []).join(', ');
        document.getElementById('manual-blue-teams').value = (blue || []).join(', ');
    }

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

    function renderCalibrationControls() {
        if (!perspectiveConfig || !perspectiveConfig.views) return;
        Object.keys(calibrationColors).forEach(viewName => {
            const view = perspectiveConfig.views[viewName];
            const count = view && Array.isArray(view.source_points) ? view.source_points.length : 0;
            const countEl = document.getElementById(`calibration-count-${viewName}`);
            const button = document.querySelector(`.calibration-view[data-view="${viewName}"]`);
            if (countEl) countEl.textContent = `${Math.min(4, count)}/4`;
            if (button) {
                button.classList.toggle('active', activeCalibrationView === viewName);
                button.classList.toggle('complete', count === 4);
            }
        });
    }

    function setCalibrationStatus(message, type = 'active') {
        calibrationStatus.textContent = message;
        calibrationStatus.className = `source-status ${type}`;
    }

    function activeCalibrationLabel() {
        if (!perspectiveConfig || !perspectiveConfig.views || !activeCalibrationView) return '';
        return perspectiveConfig.views[activeCalibrationView].label || activeCalibrationView;
    }

    function nextCalibrationPointLabel(viewName) {
        if (!perspectiveConfig || !perspectiveConfig.views || !perspectiveConfig.views[viewName]) return 'next point';
        const view = perspectiveConfig.views[viewName];
        const index = Array.isArray(view.source_points) ? view.source_points.length : 0;
        const labels = Array.isArray(view.point_labels) ? view.point_labels : [];
        return labels[index] || `point ${index + 1}`;
    }

    function savePerspectiveConfig() {
        if (!perspectiveConfig) return Promise.resolve();
        setCalibrationStatus('Saving calibration...', 'active');
        return fetch('/api/set_perspective_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(perspectiveConfig)
        })
            .then(r => r.json().then(d => ({ ok: r.ok, d })))
            .then(({ ok, d }) => {
                if (!ok) throw new Error(d.error || 'Could not save calibration');
                perspectiveConfig = d.config;
                baselinePerspectiveConfig = cloneJson(d.config);
                renderCalibrationControls();
                setCalibrationStatus('Calibration saved.', 'success');
            });
    }

    function setFirstFrame(image) {
        clearFrameError();
        const img = document.getElementById('first-frame');
        img.src = image;
        img.onload = () => { imgNativeWidth = img.naturalWidth; imgNativeHeight = img.naturalHeight; resizeCanvas(); };
    }

    function setFrameSeconds(seconds) { frameSecondsInput.value = Math.max(0, Number(seconds) || 0).toFixed(1); }

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
        if (activeCalibrationView && perspectiveConfig && perspectiveConfig.views[activeCalibrationView]) {
            const view = perspectiveConfig.views[activeCalibrationView];
            view.source_points = Array.isArray(view.source_points) ? view.source_points.slice(0, 4) : [];
            if (view.source_points.length >= 4) view.source_points = [];
            view.source_points.push([Math.round(x), Math.round(y)]);
            renderCalibrationControls();
            drawPoints();
            const remaining = 4 - view.source_points.length;
            setCalibrationStatus(
                remaining > 0 ? `${activeCalibrationLabel()}: click ${nextCalibrationPointLabel(activeCalibrationView)}.` : `${activeCalibrationLabel()} landmarks set.`,
                remaining > 0 ? 'active' : 'success'
            );
            return;
        }
        points.push([Math.round(x), Math.round(y)]);
        drawPoints();
        if (points.length >= 3) startButton.disabled = false;
    });

    function drawPoints() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!imgNativeWidth || !imgNativeHeight) return;
        const scaleX = canvas.width / imgNativeWidth, scaleY = canvas.height / imgNativeHeight;
        if (perspectiveConfig && perspectiveConfig.views) {
            Object.keys(calibrationColors).forEach(viewName => {
                const view = perspectiveConfig.views[viewName];
                const viewPoints = view && Array.isArray(view.source_points) ? view.source_points : [];
                if (!viewPoints.length) return;
                ctx.beginPath();
                ctx.strokeStyle = calibrationColors[viewName];
                ctx.lineWidth = activeCalibrationView === viewName ? 3 : 2;
                viewPoints.forEach((pt, i) => {
                    const x = pt[0] * scaleX, y = pt[1] * scaleY;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                });
                if (viewPoints.length === 4) ctx.lineTo(viewPoints[0][0] * scaleX, viewPoints[0][1] * scaleY);
                ctx.stroke();
                viewPoints.forEach((pt, i) => {
                    const x = pt[0] * scaleX, y = pt[1] * scaleY;
                    ctx.beginPath();
                    ctx.fillStyle = calibrationColors[viewName];
                    ctx.arc(x, y, 5, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.fillStyle = '#0b0b0d';
                    ctx.font = '600 10px Inter, sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(String(i + 1), x, y);
                });
            });
        }
        if (points.length > 0) {
            ctx.beginPath(); ctx.strokeStyle = '#00ffff'; ctx.lineWidth = 2;
            points.forEach((pt, i) => { const x = pt[0] * scaleX, y = pt[1] * scaleY; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); ctx.fillStyle = '#ff0000'; });
            if (points.length > 2) ctx.lineTo(points[0][0] * scaleX, points[0][1] * scaleY);
            ctx.stroke();
            points.forEach(pt => { const x = pt[0] * scaleX, y = pt[1] * scaleY; ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); });
        }
    }

    document.getElementById('btn-clear').addEventListener('click', () => clearRoi());

    document.querySelectorAll('.calibration-view').forEach(button => {
        button.addEventListener('click', () => {
            activeCalibrationView = button.dataset.view;
            const view = perspectiveConfig && perspectiveConfig.views ? perspectiveConfig.views[activeCalibrationView] : null;
            if (view) view.source_points = [];
            renderCalibrationControls();
            drawPoints();
            setCalibrationStatus(`${activeCalibrationLabel()}: click ${nextCalibrationPointLabel(activeCalibrationView)}.`, 'active');
        });
    });

    document.getElementById('btn-calibration-done').addEventListener('click', () => {
        activeCalibrationView = null;
        renderCalibrationControls();
        drawPoints();
        setCalibrationStatus('ROI editing enabled.', 'active');
    });

    document.getElementById('btn-clear-calibration-view').addEventListener('click', () => {
        if (!activeCalibrationView || !perspectiveConfig || !perspectiveConfig.views[activeCalibrationView]) return;
        perspectiveConfig.views[activeCalibrationView].source_points = [];
        renderCalibrationControls();
        drawPoints();
        setCalibrationStatus(`${activeCalibrationLabel()} cleared.`, 'active');
    });

    document.getElementById('btn-reset-calibration').addEventListener('click', () => {
        if (!baselinePerspectiveConfig) return;
        perspectiveConfig = cloneJson(baselinePerspectiveConfig);
        activeCalibrationView = null;
        renderCalibrationControls();
        drawPoints();
        setCalibrationStatus('Calibration reset.', 'active');
    });

    document.getElementById('btn-save-calibration').addEventListener('click', () => {
        savePerspectiveConfig().catch(error => {
            setCalibrationStatus(error.message || 'Could not save calibration.', 'error');
        });
    });

    // ──── Back button (ROI → Source) ────
    document.getElementById('btn-back-source').addEventListener('click', () => showPhase('phase-source'));

    // ──── Frame navigation ────
    function loadFrameAtSeconds(seconds) {
        frameStatus.textContent = 'Loading frame…'; frameStatus.className = 'source-status active';
        document.getElementById('btn-load-frame').disabled = true; startButton.disabled = true;
        fetch('/api/frame_at', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ seconds }) })
            .then(r => r.json().then(d => ({ ok: r.ok, d })))
            .then(({ ok, d }) => {
                if (!ok) { frameStatus.textContent = d.error || 'Could not load frame.'; frameStatus.className = 'source-status error'; return; }
                setFrameSeconds(d.seconds); clearRoi(); setFirstFrame(d.image);
                frameStatus.textContent = `Frame at ${Number(d.seconds).toFixed(1)}s`; frameStatus.className = 'source-status success';
            })
            .catch(() => { frameStatus.textContent = 'Connection error.'; frameStatus.className = 'source-status error'; })
            .finally(() => { document.getElementById('btn-load-frame').disabled = false; });
    }

    document.getElementById('btn-load-frame').addEventListener('click', () => loadFrameAtSeconds(Number(frameSecondsInput.value) || 0));
    document.getElementById('btn-frame-back').addEventListener('click', () => loadFrameAtSeconds(Math.max(0, (Number(frameSecondsInput.value) || 0) - 5)));
    document.getElementById('btn-frame-forward').addEventListener('click', () => loadFrameAtSeconds((Number(frameSecondsInput.value) || 0) + 5));

    // ──── YouTube loader ────
    function loadYoutubeUrl(url, matchData = null) {
        if (!url) { sourceStatus.textContent = 'Paste a YouTube URL first.'; sourceStatus.className = 'source-status error'; return; }
        sourceStatus.textContent = 'Downloading video…'; sourceStatus.className = 'source-status active';
        document.getElementById('btn-load-youtube').disabled = true;
        const payload = { youtube_url: url };
        payload.red = parseTeamList(document.getElementById('manual-red-teams').value);
        payload.blue = parseTeamList(document.getElementById('manual-blue-teams').value);
        if (matchData) {
            payload.red = payload.red.length ? payload.red : (matchData.red || []);
            payload.blue = payload.blue.length ? payload.blue : (matchData.blue || []);
        }
        fetch('/api/set_video_source', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
            .then(r => r.json().then(d => ({ ok: r.ok, d })))
            .then(({ ok, d }) => {
                if (!ok) { sourceStatus.textContent = d.error || 'Could not load video.'; sourceStatus.className = 'source-status error'; return; }
                sourceStatus.textContent = 'Video loaded.'; sourceStatus.className = 'source-status success';
                advanceToRoi(d.image, d.seconds || 0);
            })
            .catch(() => { sourceStatus.textContent = 'Connection error.'; sourceStatus.className = 'source-status error'; })
            .finally(() => { document.getElementById('btn-load-youtube').disabled = false; });
    }

    document.getElementById('btn-load-youtube').addEventListener('click', () => loadYoutubeUrl(document.getElementById('youtube-url').value.trim()));

    // ──── TBA key save ────
    document.getElementById('btn-save-tba-key').addEventListener('click', () => {
        const keyInput = document.getElementById('tba-auth-key');
        const key = keyInput.value.trim();
        if (!key) { tbaStatus.textContent = 'Paste a TBA API key first.'; tbaStatus.className = 'source-status error'; return; }
        tbaStatus.textContent = 'Saving…'; tbaStatus.className = 'source-status active';
        document.getElementById('btn-save-tba-key').disabled = true;
        fetch('/api/tba/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ auth_key: key }) })
            .then(r => r.json().then(d => ({ ok: r.ok, d })))
            .then(({ ok, d }) => {
                if (!ok) { tbaStatus.textContent = d.error || 'Could not save key.'; tbaStatus.className = 'source-status error'; return; }
                keyInput.value = ''; tbaStatus.textContent = 'TBA key saved.'; tbaStatus.className = 'source-status success';
            })
            .catch(() => { tbaStatus.textContent = 'Connection error.'; tbaStatus.className = 'source-status error'; })
            .finally(() => { document.getElementById('btn-save-tba-key').disabled = false; });
    });

    // ──── Start Processing ────
    startButton.addEventListener('click', () => {
        const processSeconds = Math.max(0, Number(processSecondsInput.value) || 0);
        const trackerMode = selectedTrackerMode();
        savePerspectiveConfig()
            .then(() => fetch('/api/set_roi', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ points, process_seconds: processSeconds, tracker_mode: trackerMode })
            }))
            .then(r => r.json()).then(d => {
                if (d.success) { showPhase('phase-processing'); pollStatus(); }
            })
            .catch(error => {
                setCalibrationStatus(error.message || 'Could not start analysis.', 'error');
            });
    });

    // ──── Poll Backend ────
    function pollStatus() {
        const interval = setInterval(() => {
            fetch('/api/status').then(r => r.json()).then(d => {
                const percent = d.total_frames > 0 ? Math.min(100, Math.round((d.progress / d.total_frames) * 100)) : 0;
                document.getElementById('progress-bar').style.width = percent + '%';
                const statusSuffix = d.processing_status ? `\n${d.processing_status}` : '';
                document.getElementById('progress-text').innerText = `${percent}% (${d.progress}/${d.total_frames})${statusSuffix}`;
                if (d.current_fuel_count !== undefined) document.getElementById('live-fuel-count').innerText = d.current_fuel_count;
                if (d.robot_path_count !== undefined) document.getElementById('live-robot-path-count').innerText = d.robot_path_count;
                if (d.preview_available) {
                    const lp = document.getElementById('live-preview');
                    lp.src = '/api/live_frame?t=' + Date.now(); lp.classList.add('active');
                    document.getElementById('live-preview-placeholder').classList.add('hidden');
                }
                if (d.is_finished) {
                    clearInterval(interval);
                    showPhase('phase-playback');
                    renderAnalysisRuns(d.analysis_runs, d.active_analysis_run);
                }
            });
        }, 1000);
    }

    analysisRunSelect.addEventListener('change', () => {
        const runKey = analysisRunSelect.value;
        if (!runKey) return;
        setActiveAnalysisRun(runKey).catch(error => {
            analysisRunStatus.className = 'source-status error';
            analysisRunStatus.textContent = error.message || 'Could not switch analysis run.';
        });
    });

    // ──── Attribution ────
    document.getElementById('btn-continue-attribution').addEventListener('click', () => {
        showPhase('phase-results');
        fetch('/api/results').then(r => r.json()).then(d => {
            const grid = document.getElementById('robot-grid');
            grid.innerHTML = '';
            const status = document.getElementById('ocr-status');
            status.className = d.ocr_status && d.ocr_status.enabled ? 'source-status success' : 'source-status active';
            status.textContent = d.ocr_status ? d.ocr_status.message : 'OCR status unavailable.';
            Object.keys(d.crops).forEach(r_id => {
                const ocr = (d.ocr_assignments || {})[r_id] || {};
                const suggestedTeam = ocr.team || '';
                const ocrText = ocr.text ? `OCR ${esc(ocr.text)}` : 'No OCR text';
                const candidate = ocr.candidate_team ? ` · closest ${ocr.candidate_team}` : '';
                const score = ocr.score !== undefined ? ` · match ${ocr.score}` : '';
                const alliance = ocr.alliance ? ` · ${esc(ocr.alliance)}` : '';
                const div = document.createElement('div'); div.className = 'robot-card';
                div.innerHTML = `<img src="data:image/jpeg;base64,${d.crops[r_id]}" /><p style="margin:0 0 8px 0; color:var(--text-muted); font-size:0.82rem">ID ${r_id}${alliance}</p><p style="margin:0 0 8px 0; color:var(--accent); font-weight:600; font-size:0.9rem">${d.scores[r_id] || 0} scored</p><p class="ocr-meta">${ocrText}${candidate}${score}</p><input type="number" class="input-field" data-id="${r_id}" placeholder="Team #" value="${suggestedTeam}" />`;
                grid.appendChild(div);
            });
        });
    });

    // ──── Chart renderer ────
    const CHART_COLORS = ['#3b82f6','#ef4444','#22c55e','#f59e0b','#a855f7','#ec4899','#14b8a6','#f97316'];

    function drawScoreChart(timeline) {
        const canvas = document.getElementById('score-chart');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        const W = rect.width - 32;  // account for container padding
        const H = 220;
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        canvas.style.width = W + 'px';
        canvas.style.height = H + 'px';
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);

        if (!timeline || !timeline.length) {
            ctx.fillStyle = '#71717a';
            ctx.font = '13px Inter, system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No scoring events recorded', W / 2, H / 2);
            return;
        }

        const pad = { top: 16, right: 16, bottom: 36, left: 40 };
        const cw = W - pad.left - pad.right;
        const ch = H - pad.top - pad.bottom;

        const maxTime = timeline[timeline.length - 1].time || 1;
        const maxScore = timeline[timeline.length - 1].cumulative || 1;

        // Collect unique teams
        const teams = [...new Set(timeline.map(e => e.team))];

        // Build per-team series as step data: [{time, total}]
        const series = {};
        teams.forEach(t => { series[t] = [{ time: 0, total: 0 }]; });
        timeline.forEach(e => { series[e.team].push({ time: e.time, total: e.team_total }); });
        // Extend each series to maxTime
        teams.forEach(t => {
            const s = series[t];
            s.push({ time: maxTime, total: s[s.length - 1].total });
        });

        function x(t) { return pad.left + (t / maxTime) * cw; }
        function y(v) { return pad.top + ch - (v / maxScore) * ch; }

        // Grid lines
        ctx.strokeStyle = '#27272a';
        ctx.lineWidth = 1;
        const yTicks = Math.min(maxScore, 6);
        for (let i = 0; i <= yTicks; i++) {
            const val = Math.round((maxScore / yTicks) * i);
            const yy = y(val);
            ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(W - pad.right, yy); ctx.stroke();
            ctx.fillStyle = '#71717a';
            ctx.font = '11px Inter, system-ui, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(val, pad.left - 6, yy + 4);
        }

        // X axis labels
        const xTicks = 5;
        ctx.textAlign = 'center';
        ctx.fillStyle = '#71717a';
        for (let i = 0; i <= xTicks; i++) {
            const t = (maxTime / xTicks) * i;
            const xx = x(t);
            ctx.fillText(t.toFixed(0) + 's', xx, H - pad.bottom + 18);
            ctx.beginPath(); ctx.moveTo(xx, pad.top); ctx.lineTo(xx, pad.top + ch); ctx.stroke();
        }

        // Draw cumulative line (background)
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x(0), y(0));
        timeline.forEach(e => { ctx.lineTo(x(e.time), y(e.cumulative - 1)); ctx.lineTo(x(e.time), y(e.cumulative)); });
        ctx.lineTo(x(maxTime), y(timeline[timeline.length - 1].cumulative));
        ctx.stroke();

        // Draw per-team step lines
        teams.forEach((team, i) => {
            const color = CHART_COLORS[i % CHART_COLORS.length];
            const pts = series[team];
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            pts.forEach((p, j) => {
                if (j === 0) { ctx.moveTo(x(p.time), y(p.total)); return; }
                ctx.lineTo(x(p.time), y(pts[j - 1].total));  // horizontal
                ctx.lineTo(x(p.time), y(p.total));            // vertical step
            });
            ctx.stroke();

            // Score dots
            ctx.fillStyle = color;
            pts.forEach((p, j) => {
                if (j === 0 || j === pts.length - 1) return;
                ctx.beginPath(); ctx.arc(x(p.time), y(p.total), 3, 0, Math.PI * 2); ctx.fill();
            });
        });

        // Legend
        const legendY = H - 8;
        let legendX = pad.left;
        ctx.font = '11px Inter, system-ui, sans-serif';
        ctx.textAlign = 'left';
        teams.forEach((team, i) => {
            const color = CHART_COLORS[i % CHART_COLORS.length];
            ctx.fillStyle = color;
            ctx.fillRect(legendX, legendY - 8, 10, 10);
            ctx.fillStyle = '#a1a1aa';
            ctx.fillText(team, legendX + 14, legendY);
            legendX += ctx.measureText(team).width + 28;
        });
    }

    // ──── Final Report ────
    document.getElementById('btn-submit-scores').addEventListener('click', () => {
        const inputs = document.querySelectorAll('.input-field');
        const mapping = {};
        inputs.forEach(input => { if (input.value) mapping[input.dataset.id] = input.value; });
        fetch('/api/submit_attribution', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mapping }) })
            .then(r => r.json()).then(d => {
                showPhase('phase-final');
                const container = document.getElementById('final-scores-container');
                let html = '';
                Object.keys(d.final_scores).sort((a, b) => d.final_scores[b] - d.final_scores[a]).forEach(team => {
                    html += `<div class="score-row"><span>Team ${team}</span><span style="color:var(--accent); font-weight:600">${d.final_scores[team]} fuel</span></div>`;
                });
                html += `<div class="score-row" style="opacity:0.7"><span>Unattributed / Manual</span><span>${d.unattributed}</span></div>`;
                container.innerHTML = html;

                // Render scoring timeline chart
                requestAnimationFrame(() => drawScoreChart(d.score_timeline));
            });
    });
});
