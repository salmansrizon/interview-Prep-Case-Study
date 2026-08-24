/* Interactive softmax lab — reusable component.
 *
 * Drag temperature and top-p, watch the distribution, entropy and nucleus cut
 * respond live. Reading the formula teaches fluency; moving the knob and
 * predicting what happens builds storage strength.
 *
 * Markup contract:
 *   <div class="softmax-lab"
 *        data-tokens="১৯৭১,১৯৫২,১৯৪৭,১৯৯০,সাল"
 *        data-logits="8.2,2.1,1.8,0.4,1.1"></div>
 *
 * No dependencies. Renders its own controls and bars.
 */
(function () {
  function softmax(logits, T) {
    var z = logits.map(function (v) { return v / T; });
    var m = Math.max.apply(null, z);
    var e = z.map(function (v) { return Math.exp(v - m); });
    var s = e.reduce(function (a, b) { return a + b; }, 0);
    return e.map(function (v) { return v / s; });
  }

  function entropy(p) {
    var h = 0;
    p.forEach(function (v) { h -= v * Math.log(v + 1e-12); });
    return Math.max(h, 0);
  }

  /* Indices kept by nucleus sampling, in descending probability order. */
  function nucleus(p, thresh) {
    var order = p.map(function (v, i) { return i; })
                 .sort(function (a, b) { return p[b] - p[a]; });
    var cum = 0, keep = [];
    for (var i = 0; i < order.length; i++) {
      keep.push(order[i]);
      cum += p[order[i]];
      if (cum >= thresh) break;
    }
    return keep;
  }

  function build(root) {
    var tokens = (root.dataset.tokens || '').split(',');
    var logits = (root.dataset.logits || '').split(',').map(Number);

    root.innerHTML =
      '<div class="lab-controls">' +
        '<label>Temperature <output class="v-t">1.00</output>' +
          '<input type="range" class="s-t" min="0.1" max="2.5" step="0.05" value="1"></label>' +
        '<label>Top-p <output class="v-p">0.90</output>' +
          '<input type="range" class="s-p" min="0.1" max="1" step="0.05" value="0.9"></label>' +
      '</div>' +
      '<div class="lab-bars"></div>' +
      '<div class="lab-readout"></div>';

    var sT = root.querySelector('.s-t');
    var sP = root.querySelector('.s-p');
    var vT = root.querySelector('.v-t');
    var vP = root.querySelector('.v-p');
    var bars = root.querySelector('.lab-bars');
    var readout = root.querySelector('.lab-readout');

    function render() {
      var T = parseFloat(sT.value);
      var P = parseFloat(sP.value);
      vT.textContent = T.toFixed(2);
      vP.textContent = P.toFixed(2);

      var p = softmax(logits, T);
      var keep = nucleus(p, P);
      var kept = {};
      keep.forEach(function (i) { kept[i] = true; });

      bars.innerHTML = tokens.map(function (tok, i) {
        var pct = (p[i] * 100).toFixed(1);
        var cls = kept[i] ? 'bar keep' : 'bar drop';
        return '<div class="bar-row">' +
                 '<span class="bar-label">' + tok + '</span>' +
                 '<span class="bar-track"><span class="' + cls +
                   '" style="width:' + Math.max(p[i] * 100, 0.4) + '%"></span></span>' +
                 '<span class="bar-val">' + pct + '%</span>' +
               '</div>';
      }).join('');

      var H = entropy(p);
      var Hmax = Math.log(tokens.length);
      var verdict = H < 0.35 * Hmax
        ? '<b style="color:#1f6b3a">মডেল নিশ্চিত</b>'
        : (H < 0.7 * Hmax
            ? '<b style="color:#7a2e1e">মোটামুটি দ্বিধায়</b>'
            : '<b style="color:#a3271b">কার্যত আন্দাজ করছে</b>');

      readout.innerHTML =
        'top-1 confidence <b>' + (Math.max.apply(null, p) * 100).toFixed(1) + '%</b> &nbsp;·&nbsp; ' +
        'entropy <b>' + H.toFixed(2) + '</b> / ' + Hmax.toFixed(2) + ' nats &nbsp;·&nbsp; ' +
        'top-p রাখছে <b>' + keep.length + '/' + tokens.length + '</b> token &nbsp;·&nbsp; ' +
        verdict;
    }

    sT.addEventListener('input', render);
    sP.addEventListener('input', render);
    render();
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.softmax-lab').forEach(build);
  });
})();
