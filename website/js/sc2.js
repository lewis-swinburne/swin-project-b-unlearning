document.addEventListener('DOMContentLoaded', () => {
    const background = document.querySelector('.matrix-background');
    const binary = ['0', '1'];
    const count = 150; // Number of binary characters to create

    // Function to generate a random number within a range
    const random = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

    // Create the falling binary characters
    for (let i = 0; i < count; i++) {
        const char = document.createElement('span');
        char.textContent = binary[random(0, 1)]; // Randomly select 0 or 1
        
        // Random horizontal position (0% to 100% width)
        char.style.left = `${random(0, 100)}vw`; 
        
        // Random duration for the fall (faster is more active)
        char.style.animationDuration = `${random(10, 30)}s`; 
        
        // Random starting delay to stagger the effect
        char.style.animationDelay = `-${random(0, 30)}s`; 
        
        // Random opacity for depth effect
        char.style.opacity = random(5, 10) / 10; 

        background.appendChild(char);
    }
});