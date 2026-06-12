/* =============================================
   MedPredict AI — app.js
   Handles: form submit, loading anim, results
   ============================================= */

(function () {
  'use strict';

  /* ── Hamburger menu ──────────────────────── */
  const hamburger = document.getElementById('hamburger');
  const navLinks  = document.querySelector('.nav-links');
  if (hamburger && navLinks) {
    hamburger.addEventListener('click', () => {
      const open = navLinks.style.display === 'flex';
      navLinks.style.cssText = open
        ? ''
        : 'display:flex;flex-direction:column;position:absolute;top:68px;left:0;right:0;background:white;padding:12px 24px 24px;border-bottom:1px solid rgba(26,111,196,.12);gap:4px;z-index:99';
    });
  }

  /* ── Form + prediction logic ─────────────── */
  const form        = document.getElementById('predictionForm');
  const predictBtn  = document.getElementById('predictBtn');
  const btnText     = document.querySelector('.btn-text');
  const btnLoading  = document.getElementById('btnLoading');
  const resultCard  = document.getElementById('resultCard');
  const resultReset = document.getElementById('resultReset');

  if (!form) return;

  form.addEventListener('submit', function (e) {
    e.preventDefault(); // Prevent default for demo; remove for Flask

    // Validate
    const inputs = form.querySelectorAll('input[required], select[required]');
    let valid = true;
    inputs.forEach(el => {
      el.style.borderColor = '';
      if (!el.value || el.value === '') {
        el.style.borderColor = '#e53e3e';
        valid = false;
      }
    });
    if (!valid) {
      shakeBtn();
      return;
    }

    // Show loading
    btnText.style.display    = 'none';
    btnLoading.style.display = 'flex';
    predictBtn.disabled      = true;
    resultCard.style.display = 'none';

    // Collect values for scoring
    const bmi       = parseFloat(form.bmi.value) || 0;
    const age       = parseInt(form.age.value)   || 0;
    const highBP    = parseInt(form.HighBP.value);
    const highChol  = parseInt(form.HighChol.value);
    const smoker    = parseInt(form.Smoker.value);
    const phys      = parseInt(form.PhysActivity.value);
    const genHlth   = parseInt(form.GenHlth.value);
    const heart     = parseInt(form.HeartDiseaseorAttack.value);

    // Simulate ML delay (replace with fetch('/predict') for Flask)
    setTimeout(() => {
      // Heuristic risk score (0–1) — replace with server response in production
      let risk = 0;
      if (bmi >= 30) risk += 0.2;
      else if (bmi >= 25) risk += 0.1;
      if (age >= 60) risk += 0.2;
      else if (age >= 45) risk += 0.12;
      if (highBP  === 1) risk += 0.15;
      if (highChol=== 1) risk += 0.12;
      if (smoker  === 1) risk += 0.08;
      if (phys    === 0) risk += 0.1;
      if (genHlth >= 4) risk += 0.1;
      if (heart   === 1) risk += 0.15;
      risk = Math.min(risk, 0.98);

      showResult(risk);

      // Restore button
      btnText.style.display    = 'flex';
      btnLoading.style.display = 'none';
      predictBtn.disabled      = false;
    }, 2200);
  });

  function showResult(risk) {
    const isHigh = risk >= 0.40;
    resultCard.className = 'result-card ' + (isHigh ? 'high' : 'low');

    document.getElementById('resultIcon').innerHTML = isHigh
      ? '<svg width="36" height="36" viewBox="0 0 36 36" fill="none"><path d="M18 3C9.72 3 3 9.72 3 18s6.72 15 15 15 15-6.72 15-15S26.28 3 18 3zm1.5 22.5h-3V22h3v3.5zm0-7h-3V10h3v8.5z" fill="#e53e3e"/></svg>'
      : '<svg width="36" height="36" viewBox="0 0 36 36" fill="none"><path d="M18 3C9.72 3 3 9.72 3 18s6.72 15 15 15 15-6.72 15-15S26.28 3 18 3zm-2 21l-5.25-5.25 2.12-2.12L16 19.75l7.13-7.13 2.12 2.13L16 24z" fill="#38a169"/></svg>';

    document.getElementById('resultTag').textContent    = isHigh ? 'ELEVATED RISK DETECTED' : 'LOW RISK DETECTED';
    document.getElementById('resultTitle').textContent  = isHigh ? 'Higher Diabetes Risk' : 'Low Diabetes Risk';
    document.getElementById('resultDesc').textContent   = isHigh
      ? `Based on your health profile, our model estimates a ${Math.round(risk * 100)}% risk score. We strongly recommend scheduling a consultation with your healthcare provider for professional evaluation and personalized guidance.`
      : `Your health indicators suggest a ${Math.round(risk * 100)}% risk score — within a healthy range. Continue maintaining your healthy habits and schedule regular check-ups as a precaution.`;

    // Risk bars
    const bars = document.getElementById('resultBars');
    bars.innerHTML = '';
    const total = 10;
    const filled = Math.round(risk * total);
    for (let i = 0; i < total; i++) {
      const span = document.createElement('span');
      if (i < filled) {
        span.style.background = isHigh
          ? `rgba(229,62,62,${0.35 + (i / total) * 0.65})`
          : `rgba(56,161,105,${0.35 + (i / total) * 0.65})`;
      } else {
        span.style.background = 'rgba(0,0,0,0.08)';
      }
      bars.appendChild(span);
    }

    resultCard.style.display = 'flex';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /* Reset button */
  if (resultReset) {
    resultReset.addEventListener('click', () => {
      resultCard.style.display = 'none';
      form.reset();
      form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }

  /* Shake animation on invalid submit */
  function shakeBtn() {
    predictBtn.style.animation = 'none';
    predictBtn.offsetHeight; // reflow
    predictBtn.style.animation = 'shakebtn 0.4s ease';
    const style = document.createElement('style');
    style.textContent = `@keyframes shakebtn {
      0%,100%{transform:translateX(0)}
      20%{transform:translateX(-8px)}
      40%{transform:translateX(8px)}
      60%{transform:translateX(-6px)}
      80%{transform:translateX(6px)}
    }`;
    if (!document.getElementById('shakeStyle')) {
      style.id = 'shakeStyle';
      document.head.appendChild(style);
    }
  }

  /* ── Input live feedback ─────────────────── */
  form.querySelectorAll('input[required]').forEach(input => {
    input.addEventListener('input', () => {
      input.style.borderColor = '';
    });
  });
  form.querySelectorAll('select[required]').forEach(sel => {
    sel.addEventListener('change', () => {
      sel.style.borderColor = '';
    });
  });

  /* ── Smooth entrance animations ─────────── */
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.field-group').forEach((el, i) => {
    el.style.opacity    = '0';
    el.style.transform  = 'translateY(18px)';
    el.style.transition = `opacity 0.45s ease ${i * 0.06}s, transform 0.45s ease ${i * 0.06}s`;
    observer.observe(el);
  });

  document.querySelectorAll('.info-card').forEach((el, i) => {
    el.style.opacity    = '0';
    el.style.transform  = 'translateY(20px)';
    el.style.transition = `opacity 0.5s ease ${i * 0.12}s, transform 0.5s ease ${i * 0.12}s`;
    observer.observe(el);
  });

})();
