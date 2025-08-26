document.addEventListener('DOMContentLoaded', function() {
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;
    const cards = document.querySelectorAll('.card');
    const prevButton = document.getElementById('prev-card');
    const nextButton = document.getElementById('next-card');
    const numberOfCards = cards.length;
    let currentIndex = 0;

    function updateCardIndices() {
        cards.forEach((card, index) => {
            const newIndex = (index - currentIndex + numberOfCards) % numberOfCards;
            card.dataset.index = newIndex;
        });
    }

    updateCardIndices();

    nextButton.addEventListener('click', () => {
        currentIndex = (currentIndex - 1 + numberOfCards) % numberOfCards;
        updateCardIndices();
    });

    prevButton.addEventListener('click', () => {
        currentIndex = (currentIndex + 1) % numberOfCards;
        updateCardIndices();
    });

    // === THEME ===
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            body.classList.toggle('light-theme');
            localStorage.setItem('theme', body.classList.contains('light-theme') ? 'light' : 'dark');
        });
        if (localStorage.getItem('theme') === 'light') {
            body.classList.add('light-theme');
        }
    }

    // === LOADER ===
    document.querySelectorAll('.card form').forEach(form => {
        form.addEventListener('submit', function(e) {
            const loader = form.querySelector('.loader-overlay');
            loader.style.display = 'flex';
            loader.style.zIndex = '1001'; // Toujours au-dessus
        });
    });
});
