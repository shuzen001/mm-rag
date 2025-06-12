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
    const res = await fetch('/mm_rag/processing-status', {
        headers: { authorization: token }
    });
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.querySelector('#files-table tbody');
    tbody.innerHTML = '';
    Object.entries(data.documents).forEach(([id, info]) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${info.filename}</td><td>${info.status}</td><td>${info.message}</td>`;
        tbody.appendChild(tr);
    });
}

setInterval(() => {
    if (token) refreshFiles();
}, 5000);
