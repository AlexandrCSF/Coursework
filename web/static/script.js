document.getElementById('searchButton').addEventListener('click', function() {
    const query = document.getElementById('searchInput').value;
    const model = document.getElementById('modelSelect').value;
    if (query) {
        searchProducts(query, model);
    }
});

async function searchProducts(query, model) {
    const url = 'http://127.0.0.1:5000/search';
    const payload = { 
        query: query,
        model: model
    };

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

        const data = await response.json();
        displayResults(data.hits.hits);
    } catch (error) {
        console.error('Error:', error);
    }
}

function displayResults(results) {
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = '';

    if (results.length === 0) {
        resultsContainer.innerHTML = '<p class="no-results">Ничего не найдено.</p>';
        return;
    }

    results.forEach(result => {
        const { name, description, picture } = result._source;
        const imageUrl = 'https://thumb.ac-illust.com/b1/b170870007dfa419295d949814474ab2_t.jpeg';
        console.log(imageUrl)
        const item = document.createElement('div');
        item.className = 'result-item';

        item.innerHTML = `
            <div class="result-card">
                <img src="${imageUrl}" alt="${name}" width="150" height="80" />
                <div class="result-info">
                    <h3 class="result-title">${name}</h3>
                    <p class="result-description">${description}</p>
                </div>
            </div>
        `;

        resultsContainer.appendChild(item);
    });
}