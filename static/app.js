let points = [];
let imgNativeWidth = 0;
let imgNativeHeight = 0;

document.addEventListener('DOMContentLoaded', () => {
    // 1. Fetch First Frame for ROI Drawing
    fetch('/api/first_frame')
        .then(res => res.json())
        .then(data => {
            if(data.image) {
                const img = document.getElementById('first-frame');
                img.src = data.image;
                img.onload = () => {
                    imgNativeWidth = img.naturalWidth;
                    imgNativeHeight = img.naturalHeight;
                    resizeCanvas();
                }
            }
        });

    const canvas = document.getElementById('roi-canvas');
    const ctx = canvas.getContext('2d');
    
    function resizeCanvas() {
        const img = document.getElementById('first-frame');
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        drawPoints();
    }
    window.addEventListener('resize', resizeCanvas);

    canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = imgNativeWidth / canvas.width;
        const scaleY = imgNativeHeight / canvas.height;
        
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        points.push([Math.round(x), Math.round(y)]);
        drawPoints();
        
        if(points.length >= 3) {
            document.getElementById('btn-start').disabled = false;
        }
    });

    function drawPoints() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const scaleX = canvas.width / imgNativeWidth;
        const scaleY = canvas.height / imgNativeHeight;

        if (points.length > 0) {
            ctx.beginPath();
            ctx.strokeStyle = '#00ffff';
            ctx.lineWidth = 2;
            
            points.forEach((pt, i) => {
                const x = pt[0] * scaleX;
                const y = pt[1] * scaleY;
                
                if(i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
                
                ctx.fillStyle = '#ff0000';
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fill();
                ctx.beginPath();
                ctx.moveTo(x, y);
            });
            
            if(points.length > 2) {
                ctx.lineTo(points[0][0] * scaleX, points[0][1] * scaleY);
            }
            ctx.stroke();
        }
    }

    document.getElementById('btn-clear').addEventListener('click', () => {
        points = [];
        drawPoints();
        document.getElementById('btn-start').disabled = true;
    });

    // 2. Start Processing
    document.getElementById('btn-start').addEventListener('click', () => {
        fetch('/api/set_roi', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({points: points})
        }).then(res => res.json()).then(data => {
            if(data.success) {
                document.getElementById('phase-roi').classList.remove('active');
                document.getElementById('phase-processing').classList.add('active');
                pollStatus();
            }
        });
    });

    // 3. Poll Backend for Completion Status
    function pollStatus() {
        const interval = setInterval(() => {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    // Update progress bar
                    const percent = data.total_frames > 0 ? Math.min(100, Math.round((data.progress / data.total_frames) * 100)) : 0;
                    document.getElementById('progress-bar').style.width = percent + '%';
                    document.getElementById('progress-text').innerText = percent + '%';

                    if(data.is_finished) {
                        clearInterval(interval);
                        
                        document.getElementById('phase-processing').classList.remove('active');
                        document.getElementById('phase-playback').classList.add('active');
                        
                        document.getElementById('video-source').src = '/static/output.mp4?t=' + new Date().getTime();
                        document.getElementById('video-player').load();
                    }
                });
        }, 1000); // Check every second
    }

    // 4. Continue to attribution
    document.getElementById('btn-continue-attribution').addEventListener('click', () => {
        showResults();
    });

    // 5. Show Robot Attribution Screen
    function showResults() {
        document.getElementById('phase-playback').classList.remove('active');
        document.getElementById('phase-results').classList.add('active');
        
        fetch('/api/results')
            .then(res => res.json())
            .then(data => {
                const grid = document.getElementById('robot-grid');
                Object.keys(data.crops).forEach(r_id => {
                    const div = document.createElement('div');
                    div.className = 'robot-card';
                    div.innerHTML = `
                        <img src="data:image/jpeg;base64,${data.crops[r_id]}" />
                        <p style="margin:0 0 10px 0; color:#94a3b8">Tracker ID: ${r_id}</p>
                        <p style="margin:0 0 10px 0; color:#ec4899; font-weight:bold">${data.scores[r_id] || 0} Points Scored</p>
                        <input type="number" class="input-field" data-id="${r_id}" placeholder="Enter Team #" />
                    `;
                    grid.appendChild(div);
                });
            });
    }

    // 6. Submit Scores and Show Final Report
    document.getElementById('btn-submit-scores').addEventListener('click', () => {
        const inputs = document.querySelectorAll('.input-field');
        const mapping = {};
        inputs.forEach(input => {
            if(input.value) {
                mapping[input.dataset.id] = input.value;
            }
        });

        fetch('/api/submit_attribution', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({mapping: mapping})
        }).then(res => res.json()).then(data => {
            document.getElementById('phase-results').classList.remove('active');
            document.getElementById('phase-final').classList.add('active');
            
            const container = document.getElementById('final-scores-container');
            let html = '';
            
            // Sort teams by highest score
            const sortedTeams = Object.keys(data.final_scores).sort((a,b) => data.final_scores[b] - data.final_scores[a]);
            
            sortedTeams.forEach(team => {
                html += `<div class="score-row"><span>Team ${team}</span><span style="color:var(--accent); font-weight:bold">${data.final_scores[team]} FUEL</span></div>`;
            });
            
            html += `<div class="score-row" style="opacity:0.7"><span>Unattributed / Manual</span><span>${data.unattributed}</span></div>`;
            
            container.innerHTML = html;
        });
    });
});
