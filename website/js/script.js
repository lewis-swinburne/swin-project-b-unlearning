document.addEventListener("DOMContentLoaded", () => {
  // ---- Section switching ----
  const sideButtons = document.querySelectorAll('.side-btn');
  const sections = document.querySelectorAll('.section');

  sideButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      sideButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const sectionId = btn.dataset.section;
      sections.forEach(sec => sec.classList.remove('active'));
      document.getElementById(sectionId).classList.add('active');
    });
  });

  // ---- Gallery functionality for each section ----
  document.querySelectorAll('.gallery-container').forEach(container => {

    const items = container.querySelectorAll('.gallery img, .gallery video');
    const dotsContainer = container.querySelector('.dots');
    const leftArrow = container.querySelector('.arrow.left');
    const rightArrow = container.querySelector('.arrow.right');
    let currentIndex = 0;

    if (items[0].tagName === "IMG") {
      const baseSrc = items[0].src.split("?")[0];
      items[0].src = baseSrc + "?t=" + Date.now();
    }

    // Create dots
    items.forEach((_, i) => {
      const dot = document.createElement('span');
      dot.classList.add('dot');
      if (i === 0) dot.classList.add('active');
      dot.addEventListener('click', () => showItem(i));
      dotsContainer.appendChild(dot);
    });

    const dots = dotsContainer.querySelectorAll('.dot');

    function showItem(index) {

      // If the current active item is a video, pause it
      const current = items[currentIndex];
      if (current.tagName === "VIDEO") {
        current.pause();
        current.currentTime = 0; // reset
      }

      // Switch active
      items[currentIndex].classList.remove('active');
      dots[currentIndex].classList.remove('active');
      currentIndex = (index + items.length) % items.length;
      items[currentIndex].classList.add('active');
      dots[currentIndex].classList.add('active');

      const activeItem = items[currentIndex];
      if (activeItem.tagName === "IMG") {
        const baseSrc = activeItem.src.split("?")[0];
        activeItem.src = baseSrc + "?t=" + Date.now();
      }
    }

    sideButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        sideButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const sectionId = btn.dataset.section;
        sections.forEach(sec => sec.classList.remove('active'));
        const newSection = document.getElementById(sectionId);
        newSection.classList.add('active');

        // Reload all GIFs in this section
        newSection.querySelectorAll('img').forEach(img => {
          if (img.src.endsWith('.gif') || img.src.includes('.gif?')) {
            const baseSrc = img.src.split("?")[0];
            img.src = baseSrc + "?t=" + Date.now();
      }
    });
  });
});

    leftArrow.addEventListener('click', () => showItem(currentIndex - 1));
    rightArrow.addEventListener('click', () => showItem(currentIndex + 1));
  });
});
