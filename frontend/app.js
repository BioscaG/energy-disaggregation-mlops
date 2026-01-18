const API_BASE = "http://127.0.0.1:8000";

async function checkHealth() {
  const r = await fetch(`${API_BASE}/health`);
  const data = await r.json();
  document.getElementById("health").textContent =
    JSON.stringify(data, null, 2);
}

async function predict(useOnnx) {
  const raw = document.getElementById("input").value;

  let x;
  try {
    x = JSON.parse(raw);
  } catch (e) {
    alert("Invalid JSON input");
    return;
  }

  const endpoint = useOnnx ? "/predict/onnx" : "/predict";

  const r = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x }),
  });

  const text = await r.text();
  document.getElementById("output").textContent = text;
}
