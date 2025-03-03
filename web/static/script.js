document.getElementById('searchButton').addEventListener('click', function() {
    const query = document.getElementById('searchInput').value;
    if (query) {
        searchProducts(query);
    }
});
async function searchProducts(query) {
    const url = 'http://127.0.0.1:5000/search';
    const payload = { query: query };

    console.log(payload);

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json(); // Parse the JSON response
        displayResults(data.hits.hits); // Display the results
    } catch (error) {
        console.error('Error:', error);
    }
}

function displayResults(results) {
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = '';

    if (results.length === 0) {
        resultsContainer.innerHTML = '<p>Ничего не найдено.</p>';
        return;
    }

    results.forEach(result => {
        const item = document.createElement('div');
        item.className = 'result-item';
        item.innerHTML = `<strong>${result._source.name}</strong><br>${result._source.description}`; // Замените поля на ваши
        resultsContainer.appendChild(item);
    });
}