document.addEventListener("DOMContentLoaded", () => {
  // ---- Section switching ----
  const sideButtons = document.querySelectorAll('.side-btn');
  const sections = document.querySelectorAll('.section');

  sideButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      // Toggle active button
      sideButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Show corresponding section
      const sectionId = btn.dataset.section;
      sections.forEach(sec => sec.classList.remove('active'));
      document.getElementById(sectionId).classList.add('active');
    });
  });

  // ---- Gallery functionality for each section ----
  document.querySelectorAll('.gallery-container').forEach(container => {
    const images = container.querySelectorAll('.gallery img');
    const dotsContainer = container.querySelector('.dots');
    const leftArrow = container.querySelector('.arrow.left');
    const rightArrow = container.querySelector('.arrow.right');
    let currentIndex = 0;

    // Create dots
    images.forEach((_, i) => {
      const dot = document.createElement('span');
      dot.classList.add('dot');
      if (i === 0) dot.classList.add('active');
      dot.addEventListener('click', () => showImage(i));
      dotsContainer.appendChild(dot);
    });

    const dots = dotsContainer.querySelectorAll('.dot');

    function showImage(index) {
      images[currentIndex].classList.remove('active');
      dots[currentIndex].classList.remove('active');
      currentIndex = (index + images.length) % images.length;
      images[currentIndex].classList.add('active');
      dots[currentIndex].classList.add('active');
    }

    leftArrow.addEventListener('click', () => showImage(currentIndex - 1));
    rightArrow.addEventListener('click', () => showImage(currentIndex + 1));
  });
});
