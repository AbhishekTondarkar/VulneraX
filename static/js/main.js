document.addEventListener('DOMContentLoaded', (event) => {
    const form = document.querySelector('form');
   const codeInput = document.getElementById('code-input');
   const submitButton = document.getElementById('submit-button');
   const resultDiv = document.getElementById('result');
   const loadingSpinner = document.getElementById('loading-spinner');
   
   form.addEventListener('submit', async (e) => {
   e.preventDefault();
   loadingSpinner.style.display = 'block';
   submitButton.disabled = true;
   resultDiv.innerHTML = '';
   
   try {
   const response = await fetch('/api/analyze', {
   method: 'POST',
   headers: {
   'Content-Type': 'application/json',
    },
   body: JSON.stringify({ code: codeInput.value }),
    });
   
   if (!response.ok) {
   throw new Error(`HTTP error! status: ${response.status}`);
    }
   
   const result = await response.json();
   
   resultDiv.innerHTML = `
    <h2>Analysis Result</h2>
    <div class="result">
    <p><strong>Vulnerable:</strong> ${result.is_vulnerable ? 'Yes' : 'No'}</p>
    <p><strong>Vulnerability Type:</strong> ${result.vulnerability_type}</p>
    <p><strong>Severity:</strong> ${result.severity}</p>
    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
    <p><strong>Repair Suggestion:</strong> ${result.repair_suggestion}</p>
    </div>
    `;
    } catch (error) {
   console.error('Error:', error);
   resultDiv.innerHTML = `<p class="error">An error occurred: ${error.message}</p>`;
    } finally {
   loadingSpinner.style.display = 'none';
   submitButton.disabled = false;
    }
    });
   });
