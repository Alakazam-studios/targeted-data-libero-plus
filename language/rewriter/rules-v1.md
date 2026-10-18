# Rewrite rules, level 1 (draft v1)

ABOUTME: The level-1 rulebook the open rewriters follow: written from Louis's categories, the robot's training
ABOUTME: vocabulary and published patterns, never from a test instruction. Tuned only on held-out paraphrases.

Provenance, so the test-independence can be checked:

- Louis's categories (2026-10-15, RESEARCH.md section 11): direct naming of described objects, common words for
  synonyms, colour handling, the other perturbations.
- The robot's training vocabulary: the nouns, attributes and verbs of the 40 training instructions
  (`runs/xp/fairlang/paraphrases/training-tasks.jsonl`, byte-identical to the training data's `meta/tasks.jsonl`).
  A word list, never the 40 sentences.
- Published patterns, written here in our own words: Watts & Cui, arXiv 2610.10526 (lexical swaps measured on π0.5,
  "an added color only ever lowers success", keep selectors and step structure); the kinds of perturbation named in
  Hugo's closed-set prompt (objects described by appearance, material or function; padded; indirect).
- Not used: any LIBERO-Plus test instruction. The author of this file has not read the language-axis test set.

## The vocabulary (given to the rewriter in the vocabulary-aware prompt only)

Objects and fixtures: alphabet soup, basket, bbq sauce, black bowl, bowl, book, butter, cabinet, caddy (its back
compartment), chocolate pudding, cookie box, cream cheese, cream cheese box, drawer (top, middle, bottom), ketchup,
microwave, milk, moka pot, mug (white mug, yellow and white mug), orange juice, plate, rack, ramekin, salad dressing,
stove, table center, tomato sauce, wine bottle, wooden cabinet.

Actions: put ... on/in ..., pick up ... and place it on/in ..., open, close, turn on, push ... to ....

Positions and counts: left, right, top, middle, bottom, front, back, between, next to, on top of, inside, to the right
of, both.

## The rules

1. Output one instruction for the robot: lowercase, imperative, no final period, nothing else.
2. Delete everything that is not part of what the robot must do: greetings, politeness, reasons, stories, comments
   about the scene that do not change the action, and any trailing text that reads like a file name or parameters
   (numbers, and words such as "view", "initstate", "noise", "table" followed by numbers).
3. When the instruction states a goal, a reason chain or a riddle instead of an action, write the action that reaches
   it ("the bowl belongs on the plate, see to it" becomes "put the bowl on the plate").
4. Name every object directly. When an object is described by its appearance, material, use, contents, or by a
   riddle, replace the description with the plain name of the one object it denotes (in the vocabulary-aware prompt,
   a name from the vocabulary; with an image, the object the image shows). If more than one object could fit, keep the
   words as written.
5. Keep a locating phrase that picks out one object among several with the same name ("the black bowl next to the
   ramekin", "the left plate", "the bowl in the top drawer"). Rewrite its landmark with a plain name, but never drop it
   and never add one.
6. Replace a less common synonym with the common word the robot knows: a cooking appliance (hob, hot plate, burner,
   cooktop, range) is "the stove"; a flat dish for food is "the plate"; a drinking cup with a handle is "the mug";
   cupboard is "cabinet"; to grab or lift is "pick up"; to switch on is "turn on"; to shut is "close". Never replace a
   specific name with a broader one ("ketchup" never becomes "sauce" or "bottle").
7. Keep a colour or material only when it is part of the object's name ("black bowl", "white mug", "yellow and white
   mug", "wooden cabinet") or when it tells two objects apart. Never add a colour, material, size or shape that the
   instruction does not need.
8. Keep every step, in order, with its pronouns ("... and close it"), every count word ("both") and every position
   word. Never merge two steps or split one.
9. If a part of the instruction is unclear, keep its original words rather than guess.
