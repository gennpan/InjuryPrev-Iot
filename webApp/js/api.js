async function getRiskPrediction(sequence) {
    const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence })
    });

    const data = await response.json();
    return data.risk_probability;
}
