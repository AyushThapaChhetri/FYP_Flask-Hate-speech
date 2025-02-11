document.getElementById('predictForm').addEventListener('submit', function (e) {
    e.preventDefault();
    const inputText = document.getElementById('inputText').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
    })
        .then(response => response.json())
        .then(data => {
            const resultDiv = document.getElementById('result');
            const resultTable = document.getElementById('resultTable');
            const tbody = resultTable.querySelector('tbody');

            // Clear previous results
            tbody.innerHTML = '';

            let predictionText = '';
            let predictionColor = '';

            switch (data.prediction) {
                case 0:
                    predictionText = 'Hate Speech';
                    predictionColor = 'red';
                    break;
                case 1:
                    predictionText = 'Offensive Language';
                    predictionColor = 'orange';
                    break;
                default:
                    predictionText = 'Neither';
                    predictionColor = 'grey';
            }

            // CHANGE: Added color styling to result div
            resultDiv.innerHTML = `Prediction: <span style="color: ${predictionColor}; font-weight: bold;">${predictionText}</span>`;


            // Display results
            // resultDiv.innerHTML = `Prediction: ${data.prediction === 0 ? 'Hate Speech' : data.prediction === 1 ? 'Offensive Language' : 'Neither'}`;
            const row = document.createElement('tr');
            row.innerHTML = `
          <td>${inputText}</td>
          <td>${data.prediction === 0 ? 'Hate Speech' : data.prediction === 1 ? 'Offensive Language' : 'Neither'}</td>
          <td>${(data.probabilities[0] * 100).toFixed(2)}% Hate, ${(data.probabilities[1] * 100).toFixed(2)}% Offensive, ${(data.probabilities[2] * 100).toFixed(2)}% Neither</td>
        `;
            tbody.appendChild(row);

            // Show the result table
            resultTable.style.display = 'table';
        })
        .catch(error => console.error('Error:', error));
});
