let savedEmbeddingsData = [];
let allEmployees = [];
let currentRosterFilter = 'all';
let rosterSearchQuery = '';
let lastSeenAttendanceTimes = {};

document.addEventListener('DOMContentLoaded', () => {
  initDashboard();
  setInterval(refreshDashboardData, 2500);
});

async function initDashboard() {
  await refreshDashboardData();
  await loadSavedEmbeddings();
  await loadEmployees();
}

async function refreshDashboardData() {
  try {
    // 1. Status & Metrics
    const statusRes = await fetch('/api/status');
    if (statusRes.ok) {
      const s = await statusRes.json();
      document.getElementById('stat-fps').innerText = `${s.target_fps.toFixed(1)} FPS`;
      document.getElementById('stat-total-captures').innerText = s.total_captures || 0;
      document.getElementById('stat-quality-rate').innerText = `${s.quality_pass_rate || 0}%`;
      document.getElementById('stat-enrolled-count').innerText = s.enrolled_employees || 0;
      document.getElementById('stat-attendance-count').innerText = s.total_attendance || 0;

      const statusDot = document.querySelector('#camera-status-pill .status-dot');
      const statusText = document.getElementById('camera-status-text');
      if (s.status === 'online') {
        statusDot.className = 'status-dot live';
        statusText.innerText = 'RTSP Online (Zero Lag)';
      } else {
        statusDot.className = 'status-dot';
        statusText.innerText = 'Connecting...';
      }
    }

    // 2. Attendance Table
    const attRes = await fetch('/api/attendance');
    if (attRes.ok) {
      const events = await attRes.json();
      renderAttendanceTable(events);
    }

    // 3. Captures Table
    const capRes = await fetch('/api/captures');
    if (capRes.ok) {
      const caps = await capRes.json();
      renderCapturesTable(caps);
    }

    // 4. Live Employee Roster & Attendance Status
    await loadEmployees();
  } catch (err) {
    console.error('Error refreshing dashboard:', err);
  }
}

async function loadSavedEmbeddings() {
  try {
    const res = await fetch('/api/embeddings');
    if (!res.ok) return;
    const items = await res.json();
    savedEmbeddingsData = items;

    // Render gallery
    const gallery = document.getElementById('embeddings-gallery');
    if (items.length === 0) {
      gallery.innerHTML = '<div class="empty-state">No embeddings saved yet. Faces detected through RTSP stream will appear here.</div>';
    } else {
      gallery.innerHTML = items.map(item => {
        const meta = item.quality_metrics || {};
        const imgName = item.aligned_image ? item.aligned_image.split(/[\\/]/).pop() : '';
        const imgSrc = imgName ? `/faces/${imgName}` : '';
        const blur = meta.blur_score ? meta.blur_score.toFixed(1) : '-';
        const matchName = meta.matched_name || 'UNKNOWN';

        return `
          <div class="face-card">
            <div class="face-thumb-wrapper">
              ${imgSrc ? `<img src="${imgSrc}" alt="Aligned Face" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'50\\' height=\\'50\\'><rect fill=\\'%23222\\' width=\\'50\\' height=\\'50\\'/></svg>'">` : ''}
            </div>
            <div class="face-card-info">
              <span class="face-card-name">${matchName}</span>
              <span class="face-card-meta">Blur: ${blur} | 128-d</span>
              <span class="face-card-meta" style="color: #64748b;">${item.id}</span>
            </div>
          </div>
        `;
      }).join('');
    }

    // Populate Verification Select Dropdowns
    const sel1 = document.getElementById('verify-select-1');
    const sel2 = document.getElementById('verify-select-2');

    const optsHtml = '<option value="">Select saved embedding...</option>' + items.map(item => {
      const meta = item.quality_metrics || {};
      const label = `${item.id} (${meta.matched_name || 'Unknown'}) - Blur: ${meta.blur_score ? meta.blur_score.toFixed(0) : '-'}`;
      return `<option value="${item.embedding_npy}">${label}</option>`;
    }).join('');

    const val1 = sel1.value;
    const val2 = sel2.value;
    sel1.innerHTML = optsHtml;
    sel2.innerHTML = optsHtml;
    if (val1) sel1.value = val1;
    if (val2) sel2.value = val2;

    if (!val1 && items.length >= 1) sel1.value = items[0].embedding_npy;
    if (!val2 && items.length >= 2) sel2.value = items[1].embedding_npy;
  } catch (err) {
    console.error('Error loading embeddings:', err);
  }
}

async function loadEmployees() {
  try {
    const res = await fetch('/api/employees');
    if (!res.ok) return;
    const emps = await res.json();
    allEmployees = emps;

    // Detect new attendance confirmations to trigger visual pulse/flash
    const justConfirmedIds = new Set();
    emps.forEach(emp => {
      const prevTime = lastSeenAttendanceTimes[emp.id];
      if (emp.last_attendance_time && emp.last_attendance_time !== prevTime) {
        if (prevTime !== undefined) {
          justConfirmedIds.add(emp.id);
        }
        lastSeenAttendanceTimes[emp.id] = emp.last_attendance_time;
      }
    });

    // Update badges
    const totalBadge = document.getElementById('enrolled-badge');
    const presentBadge = document.getElementById('today-present-badge');
    if (totalBadge) totalBadge.innerText = `${emps.length} Staff`;

    const presentCount = emps.filter(e => e.attended_today).length;
    if (presentBadge) {
      presentBadge.innerText = `${presentCount} Present Today`;
      presentBadge.className = presentCount > 0 ? 'badge badge-success' : 'badge';
    }

    renderEmployeeList(justConfirmedIds);
  } catch (err) {
    console.error('Error loading employees:', err);
  }
}

function renderEmployeeList(justConfirmedIds = new Set()) {
  const container = document.getElementById('employee-list');
  if (!container) return;

  // Filter employees
  let filtered = allEmployees.filter(emp => {
    // 1. Filter by status
    if (currentRosterFilter === 'confirmed' && !emp.attended_today) return false;
    if (currentRosterFilter === 'pending' && emp.attended_today) return false;

    // 2. Filter by search query
    if (rosterSearchQuery) {
      const q = rosterSearchQuery.toLowerCase();
      const matchName = (emp.name || '').toLowerCase().includes(q);
      const matchId = (emp.id || '').toLowerCase().includes(q);
      const matchDept = (emp.department || '').toLowerCase().includes(q);
      return matchName || matchId || matchDept;
    }
    return true;
  });

  if (filtered.length === 0) {
    container.innerHTML = '<div class="empty-state">No matching staff found in roster.</div>';
    return;
  }

  container.innerHTML = filtered.map(emp => {
    const isAttended = emp.attended_today === true;
    const isJustConfirmed = justConfirmedIds.has(emp.id);

    // Format readable confirmation time
    let timeText = 'Not Checked In';
    if (emp.last_attendance_time) {
      const raw = emp.last_attendance_time.replace('T', ' ');
      const parts = raw.split(' ');
      const timePart = parts[1] ? parts[1].slice(0, 8) : '';
      timeText = `Today at ${timePart}`;
      if (emp.last_confidence) {
        timeText += ` (${Math.round(emp.last_confidence * 100)}%)`;
      }
    }

    // Avatar image or clean fallback
    const avatarSrc = emp.avatar_url || '';
    const initial = (emp.name || 'S').charAt(0).toUpperCase();

    // Confirmation Sign Badge
    const signHtml = isAttended
      ? `
        <div class="attendance-confirmation-sign confirmed">
          <div class="sign-pill">
            <span class="sign-icon">✓</span>
            <span class="sign-text">CONFIRMED</span>
          </div>
          <span class="sign-time">${timeText}</span>
        </div>
      `
      : `
        <div class="attendance-confirmation-sign pending">
          <div class="sign-pill">
            <span class="sign-dot"></span>
            <span class="sign-text">Pending</span>
          </div>
          <span class="sign-time">Not Checked In</span>
        </div>
      `;

    const itemClasses = ['employee-item'];
    if (isAttended) itemClasses.push('is-confirmed');
    if (isJustConfirmed) itemClasses.push('just-confirmed');

    return `
      <div class="${itemClasses.join(' ')}" id="emp-row-${emp.id}">
        <div class="emp-main-cell">
          <img class="emp-avatar" src="${avatarSrc}" onerror="this.onerror=null; this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'44\\' height=\\'44\\'><rect fill=\\'%23334155\\' width=\\'44\\' height=\\'44\\'/><text x=\\'50%\\' y=\\'55%\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' fill=\\'white\\' font-size=\\'18\\' font-family=\\'sans-serif\\'>${initial}</text></svg>'">
          <div class="emp-info">
            <span class="emp-name" title="${emp.name}">${emp.name}</span>
            <span class="emp-sub">ID: ${emp.id} &bull; ${emp.department || 'Staff'}</span>
          </div>
        </div>
        <div class="emp-status-cell">
          ${signHtml}
        </div>
        <div class="emp-actions">
          <button class="emp-del-btn" title="Delete staff profile" onclick="deleteStaff('${emp.id}', '${encodeURIComponent(emp.name)}')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6"/>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
          </button>
        </div>
      </div>
    `;
  }).join('');
}

function handleRosterFilter() {
  const input = document.getElementById('roster-search-input');
  rosterSearchQuery = input ? input.value.trim() : '';
  renderEmployeeList();
}

function setRosterFilter(filter) {
  currentRosterFilter = filter;
  document.querySelectorAll('.filter-pill').forEach(pill => pill.classList.remove('active'));
  const activePill = document.getElementById(`filter-pill-${filter}`);
  if (activePill) activePill.classList.add('active');
  renderEmployeeList();
}

async function deleteStaff(empId, encodedName) {
  const name = decodeURIComponent(encodedName);
  if (!confirm(`Are you sure you want to delete staff profile '${name}' (ID: ${empId})?`)) {
    return;
  }
  try {
    const res = await fetch(`/api/employees/${empId}`, { method: 'DELETE' });
    const data = await res.json();
    if (res.ok) {
      await loadEmployees();
      await loadSavedEmbeddings();
    } else {
      alert(data.error || 'Failed to delete staff profile.');
    }
  } catch (err) {
    alert('Network error while deleting staff.');
  }
}

function renderAttendanceTable(events) {
  const tbody = document.getElementById('attendance-tbody');
  if (!events || events.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="text-center">No attendance events recorded yet.</td></tr>';
    return;
  }

  tbody.innerHTML = events.map(ev => {
    const timeFormatted = ev.timestamp ? ev.timestamp.replace('T', ' ').slice(0, 19) : '-';
    const photoName = ev.face_image_path ? ev.face_image_path.split(/[\\/]/).pop() : '';
    const photoSrc = photoName ? `/faces/${photoName}` : '';
    const embName = ev.embedding_path ? ev.embedding_path.split(/[\\/]/).pop() : '-';

    return `
      <tr>
        <td>#${ev.id}</td>
        <td><strong>${ev.employee_name}</strong> <span style="color:#64748b; font-size:0.75rem;">(${ev.employee_id})</span></td>
        <td style="font-family: var(--font-mono); font-size: 0.8rem;">${timeFormatted}</td>
        <td>${ev.camera_id}</td>
        <td>${(ev.confidence * 100).toFixed(1)}%</td>
        <td><span class="status-badge confirmed">${ev.status}</span></td>
        <td>${photoSrc ? `<img src="${photoSrc}" style="width:34px;height:34px;border-radius:4px;object-fit:cover;">` : '-'}</td>
        <td style="font-family: var(--font-mono); font-size:0.72rem; color:#94a3b8;">${embName}</td>
      </tr>
    `;
  }).join('');
}

function renderCapturesTable(captures) {
  const tbody = document.getElementById('captures-tbody');
  if (!captures || captures.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-center">No capture audits recorded yet.</td></tr>';
    return;
  }

  tbody.innerHTML = captures.map(c => {
    const timeFormatted = c.timestamp ? c.timestamp.split('T')[1].slice(0, 8) : '-';
    const passed = c.passed_quality === 1;
    const badgeClass = passed ? 'confirmed' : 'rejected';
    const badgeText = passed ? 'PASSED' : 'REJECTED';
    const note = passed ? `Match: ${c.match_name} (${(c.match_score * 100).toFixed(0)}%)` : (c.rejection_reason || 'Quality failure');

    return `
      <tr>
        <td style="font-family: var(--font-mono);">${timeFormatted}</td>
        <td>${c.bbox_w ? Math.round(c.bbox_w) : '-'}x${c.bbox_h ? Math.round(c.bbox_h) : '-'}px</td>
        <td>${c.blur_score ? c.blur_score.toFixed(1) : '-'}</td>
        <td>${c.roll_angle ? c.roll_angle.toFixed(1) : '-'}°</td>
        <td>${c.yaw_ratio ? c.yaw_ratio.toFixed(2) : '-'}</td>
        <td><span class="status-badge ${badgeClass}">${badgeText}</span></td>
        <td style="font-size:0.8rem; color:${passed ? '#86efac' : '#fca5a5'};">${note}</td>
      </tr>
    `;
  }).join('');
}

async function runEmbeddingVerification() {
  const sel1 = document.getElementById('verify-select-1');
  const sel2 = document.getElementById('verify-select-2');
  const file1 = sel1.value;
  const file2 = sel2.value;

  if (!file1 || !file2) {
    alert('Please select both embeddings to verify.');
    return;
  }

  const btn = document.getElementById('run-verify-btn');
  btn.disabled = true;
  btn.innerText = 'Calculating Distance...';

  try {
    const res = await fetch('/api/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file1, file2 })
    });

    const data = await res.json();
    if (res.ok) {
      const box = document.getElementById('verify-result-box');
      box.classList.remove('hidden');

      const title = document.getElementById('verdict-title');
      if (data.is_match) {
        title.innerText = 'MATCH / SAME IDENTITY';
        title.className = 'verdict-label match';
      } else {
        title.innerText = 'NO MATCH / DIFFERENT PERSON';
        title.className = 'verdict-label no-match';
      }

      document.getElementById('verdict-sim').innerText = `${data.cosine_similarity.toFixed(4)}`;
      document.getElementById('verdict-cosine').innerText = `${data.cosine_similarity.toFixed(4)} (${data.confidence_percent}%)`;
      document.getElementById('verdict-l2').innerText = `${data.l2_distance.toFixed(4)}`;
    } else {
      alert('Verification Error: ' + (data.error || 'Failed to compare'));
    }
  } catch (err) {
    alert('Failed to connect to verification API.');
  } finally {
    btn.disabled = false;
    btn.innerText = 'Run Cosine Verification';
  }
}

function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.add('hidden'));

  document.getElementById(`tab-btn-${tabName}`).classList.add('active');
  document.getElementById(`tab-content-${tabName}`).classList.remove('hidden');

  if (tabName === 'embeddings') {
    loadSavedEmbeddings();
  }
}

// 3-Photos & Folder Preview Controls
function updateFolderPreview() {
  const idInput = document.getElementById('emp-id-input');
  const nameInput = document.getElementById('emp-name-input');
  const preview = document.getElementById('folder-preview-path');
  if (!preview) return;

  const id = idInput && idInput.value.trim() ? idInput.value.trim().replace(/[^\w\-]/g, '_') : 'staffid';
  const name = nameInput && nameInput.value.trim() ? nameInput.value.trim().replace(/[^\w\-]/g, '_') : 'name';
  preview.innerText = `Attendence System\\Staffs\\${id}_${name}`;
}

function triggerPhotoSelect(slotNum) {
  const input = document.getElementById(`photo${slotNum}-input`);
  if (input) input.click();
}

function handlePhotoSelect(event, slotNum) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = document.getElementById(`img-preview-${slotNum}`);
    const preview = document.getElementById(`preview-${slotNum}`);
    const placeholder = document.getElementById(`placeholder-${slotNum}`);
    if (img && preview && placeholder) {
      img.src = e.target.result;
      preview.classList.remove('hidden');
      placeholder.classList.add('hidden');
    }
  };
  reader.readAsDataURL(file);
}

function clearPhoto(event, slotNum) {
  event.stopPropagation();
  const input = document.getElementById(`photo${slotNum}-input`);
  const preview = document.getElementById(`preview-${slotNum}`);
  const placeholder = document.getElementById(`placeholder-${slotNum}`);
  const img = document.getElementById(`img-preview-${slotNum}`);

  if (input) input.value = '';
  if (img) img.src = '';
  if (preview) preview.classList.add('hidden');
  if (placeholder) placeholder.classList.remove('hidden');
}

function resetEnrollForm() {
  const form = document.getElementById('enroll-form');
  if (form) form.reset();
  for (let i = 1; i <= 3; i++) {
    const preview = document.getElementById(`preview-${i}`);
    const placeholder = document.getElementById(`placeholder-${i}`);
    const img = document.getElementById(`img-preview-${i}`);
    if (img) img.src = '';
    if (preview) preview.classList.add('hidden');
    if (placeholder) placeholder.classList.remove('hidden');
  }
  updateFolderPreview();
}

// Modal controls
function openEnrollModal() {
  resetEnrollForm();
  document.getElementById('enroll-modal').classList.remove('hidden');
  document.getElementById('enroll-error').classList.add('hidden');
  document.getElementById('enroll-success').classList.add('hidden');
}

function closeEnrollModal() {
  document.getElementById('enroll-modal').classList.add('hidden');
}

async function handleEnrollSubmit(event) {
  event.preventDefault();
  const form = document.getElementById('enroll-form');
  const formData = new FormData(form);
  const btn = document.getElementById('enroll-submit-btn');
  const errorBanner = document.getElementById('enroll-error');
  const successBanner = document.getElementById('enroll-success');

  // Verify photos
  const p1 = document.getElementById('photo1-input').files[0];
  const p2 = document.getElementById('photo2-input').files[0];
  const p3 = document.getElementById('photo3-input').files[0];

  if (!p1 && !p2 && !p3) {
    errorBanner.innerText = 'Please select portrait photos before confirming registration.';
    errorBanner.classList.remove('hidden');
    return;
  }

  btn.disabled = true;
  btn.innerText = 'Extracting Embeddings & Creating Folder...';
  errorBanner.classList.add('hidden');
  successBanner.classList.add('hidden');

  try {
    const res = await fetch('/api/enroll', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (res.ok) {
      successBanner.innerText = data.message || 'Staff registered successfully!';
      successBanner.classList.remove('hidden');
      await loadEmployees();
      await loadSavedEmbeddings();
      setTimeout(closeEnrollModal, 1600);
    } else {
      errorBanner.innerText = data.error || 'Registration failed. Ensure clear face photos with good lighting.';
      errorBanner.classList.remove('hidden');
    }
  } catch (err) {
    errorBanner.innerText = 'Network error while registering staff.';
    errorBanner.classList.remove('hidden');
  } finally {
    btn.disabled = false;
    btn.innerText = 'Confirm Register';
  }
}
