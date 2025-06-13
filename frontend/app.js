let token = '';
const loginSection = document.getElementById('login-section');
const mainSection = document.getElementById('main-section');
const loginError = document.getElementById('login-error');

document.getElementById('login-btn').addEventListener('click', async () => {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    try {
        const res = await fetch('/mm_rag/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        if (!res.ok) throw new Error('Login failed');
        const data = await res.json();
        token = data.token;
        loginSection.style.display = 'none';
        mainSection.style.display = 'block';
        refreshFiles();
    } catch (err) {
        loginError.textContent = '登入失敗';
        loginError.style.display = 'block';
    }
});

document.getElementById('file-input').addEventListener('change', (e) => {
    const file = e.target.files[0];
    document.getElementById('file-name').textContent = file ? file.name : '未選擇檔案';
});

document.getElementById('upload-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('file-input');
    if (!fileInput.files.length) return;
    const form = new FormData();
    form.append('file', fileInput.files[0]);
    form.append('process_immediately', 'true');
    await fetch('/mm_rag/upload', {
        method: 'PUT',
        headers: { authorization: token },
        body: form
    });
    fileInput.value = '';
    document.getElementById('file-name').textContent = '未選擇檔案';
    refreshFiles();
});

async function refreshFiles() {
    const [statusRes, filesRes] = await Promise.all([
        fetch('/mm_rag/processing-status', { headers: { authorization: token } }),
        fetch('/mm_rag/files', { headers: { authorization: token } })
    ]);
    if (!statusRes.ok || !filesRes.ok) return;
    const statusData = await statusRes.json();
    const fileData = await filesRes.json();
    const statusMap = {};
    Object.values(statusData.documents).forEach((info) => {
        statusMap[info.filename] = { status: info.status, message: info.message };
    });

    const tbody = document.querySelector('#files-table tbody');
    tbody.innerHTML = '';
    fileData.files.forEach((name) => {
        const s = statusMap[name] || { status: 'uploaded', message: '' };
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${name}</td><td>${s.status}</td><td>${s.message || ''}</td>` +
            `<td><button class="delete-btn" data-file="${name}">刪除</button></td>`;
        tbody.appendChild(tr);
    });

    document.querySelectorAll('.delete-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const form = new FormData();
            form.append('file_name', btn.dataset.file);
            await fetch('/mm_rag/delete', {
                method: 'POST',
                headers: { authorization: token },
                body: form
            });
            refreshFiles();
        });
    });
}

setInterval(() => {
    if (token) refreshFiles();
}, 5000);
