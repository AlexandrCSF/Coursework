// Tab switching
document.querySelectorAll('.tab-button').forEach(button => {
    button.addEventListener('click', () => {
        // Remove active class from all buttons and contents
        document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        
        // Add active class to clicked button and corresponding content
        button.classList.add('active');
        document.getElementById(`${button.dataset.tab}-tab`).classList.add('active');
        
        // Load products if switching to all products tab
        if (button.dataset.tab === 'all-products') {
            loadAllProducts(currentPage);
        }
    });
});

// Dataset selection handler
document.getElementById('datasetSelect').addEventListener('change', function() {
    const modelSelect = document.getElementById('modelSelect');
    const selectedDataset = this.value;
    
    if (selectedDataset === 'wildberries') {
        // Сохраняем текущую выбранную модель
        const currentModel = modelSelect.value;
        
        // Отключаем все опции кроме multilingual
        Array.from(modelSelect.options).forEach(option => {
            option.disabled = option.value !== 'multilingual';
        });
        
        // Устанавливаем multilingual
        modelSelect.value = 'multilingual';
    } else {
        // Разблокируем все опции
        Array.from(modelSelect.options).forEach(option => {
            option.disabled = false;
        });
    }
});

// Search functionality
document.getElementById('searchButton').addEventListener('click', function() {
    const query = document.getElementById('searchInput').value;
    const model = document.getElementById('modelSelect').value;
    const dataset = document.getElementById('datasetSelect').value;
    if (query) {
        searchProducts(query, model, dataset);
    }
});

// All products functionality
let currentPage = 1;
const productsPerPage = 20;

document.getElementById('prevPage').addEventListener('click', function() {
    if (currentPage > 1) {
        currentPage--;
        loadAllProducts(currentPage);
    }
});

document.getElementById('nextPage').addEventListener('click', function() {
    currentPage++;
    loadAllProducts(currentPage);
});

async function loadAllProducts(page) {
    const url = `http://127.0.0.1:5000/all_products?page=${page}&size=${productsPerPage}`;
    
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update pagination controls
        document.getElementById('currentPage').textContent = `Страница ${page}`;
        document.getElementById('prevPage').disabled = page === 1;
        document.getElementById('nextPage').disabled = 
            data.amazon.length < productsPerPage && data.wildberries.length < productsPerPage;
        
        // Display products
        displayAllProducts(data);
    } catch (error) {
        console.error('Error:', error);
    }
}

function displayAllProducts(data) {
    // Display Amazon products
    const amazonContainer = document.getElementById('amazon-products');
    amazonContainer.innerHTML = '';
    data.amazon.forEach(product => {
        const item = createProductCard(product._source);
        amazonContainer.appendChild(item);
    });
    
    // Display Wildberries products
    const wildberriesContainer = document.getElementById('wildberries-products');
    wildberriesContainer.innerHTML = '';
    data.wildberries.forEach(product => {
        const item = createProductCard(product._source);
        wildberriesContainer.appendChild(item);
    });
}

function createProductCard(product) {
    const { name, description, picture } = product;

    const item = document.createElement('div');
    item.className = 'result-item';
    
    item.innerHTML = `
        <div class="result-card">
            <img src="${picture}" alt="${name}" width="150" height="80" />
            <div class="result-info">
                <h3 class="result-title">${name}</h3>
                <p class="result-description">${description}</p>
            </div>
        </div>
    `;
    
    return item;
}

async function searchProducts(query, model, dataset) {
    const url = 'http://127.0.0.1:5000/search';
    const payload = { 
        query: query,
        model: model,
        dataset: dataset
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
        const item = createProductCard(result._source);
        resultsContainer.appendChild(item);
    });
}