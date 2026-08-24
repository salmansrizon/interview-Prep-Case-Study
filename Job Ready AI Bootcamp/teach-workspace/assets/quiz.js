/* Reusable quiz + recall widgets. Shared by every lesson in ./lessons/.
 *
 * Markup contract:
 *   <div class="quiz" data-answer="1">
 *     <p class="q">Question text</p>
 *     <button class="opt">option a</button>
 *     <button class="opt">option b</button>
 *     <p class="why">Explanation shown after answering.</p>
 *   </div>
 *
 * data-answer is the zero-based index of the correct option.
 * Feedback is immediate and automatic — the tight loop is the point.
 */
(function () {
  function wireQuiz(quiz) {
    var correct = parseInt(quiz.dataset.answer, 10);
    var opts = Array.prototype.slice.call(quiz.querySelectorAll('button.opt'));
    var why = quiz.querySelector('.why');

    opts.forEach(function (btn, i) {
      btn.addEventListener('click', function () {
        if (quiz.dataset.done === 'true') return;
        quiz.dataset.done = 'true';

        opts.forEach(function (b, j) {
          b.disabled = true;
          if (j === correct) b.classList.add('right');
        });
        if (i !== correct) btn.classList.add('wrong');
        if (why) why.classList.add('show');
      });
    });
  }

  /* Recall boxes reveal the model answer only after the learner has
     committed something to the textarea — retrieval before feedback. */
  function wireRecall(box) {
    var area = box.querySelector('textarea');
    var reveal = box.querySelector('button.reveal');
    var answer = box.querySelector('.model-answer');
    if (!reveal || !answer) return;

    answer.style.display = 'none';
    reveal.addEventListener('click', function () {
      if (area && area.value.trim().length < 3) {
        reveal.textContent = 'আগে নিজে লিখো, তারপর মিলিয়ে দেখো';
        return;
      }
      answer.style.display = 'block';
      reveal.disabled = true;
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.quiz').forEach(wireQuiz);
    document.querySelectorAll('.recall').forEach(wireRecall);
  });
})();
