document.addEventListener('DOMContentLoaded', function() {
    const showResultsButton = document.getElementById('show-results-button');
    const detectionDetailsContainer = document.getElementById('detection-details-container');

    if (showResultsButton && detectionDetailsContainer) {
        showResultsButton.addEventListener('click', () => {
            if (detectionDetailsContainer.style.display === 'none' || detectionDetailsContainer.style.display === '') {
                detectionDetailsContainer.style.display = 'block';
                showResultsButton.textContent = 'Hide Detection Details';
            } else {
                detectionDetailsContainer.style.display = 'none';
                showResultsButton.textContent = 'Show Detection Details';
            }
        });
    }

    // Theme toggle functionality
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            body.classList.toggle('light-theme');
            const isLightTheme = body.classList.contains('light-theme');
            localStorage.setItem('theme', isLightTheme ? 'light' : 'dark');
        });

        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            body.classList.add('light-theme');
        }
    }
});