# SPE → ARC-AGI-3 Strategic Roadmap

## Why SPE has structural advantage over LLMs

LLMs on ARC-AGI-3:
- See grid as token sequence
- Pattern-match to training distribution  
- Cannot update world model mid-episode
- Score: <1% on ARC-AGI-3

SPE architecture:
- Causal graph is native representation
- det-cf engine executes interventions exactly
- HWM designed for online belief updating
- CLMG stores episodic observations associatively

## The Missing Layer: Grid → SCM Induction

Current: CLADDER gives us the SCM in text
Need:    Induce SCM from ARC grid observations

### What ARC-AGI-3 grids look like
- 2D colored grids (input → output pairs)
- Rules are deterministic and compositional
- Must generalize from 3-5 examples to novel test
- Interactive version: act, observe result, update

### Induction strategy (buildable in 4-6 weeks)
1. Grid encoder: pixel grid → object representation
   - Detect objects (connected regions of same color)
   - Extract attributes: color, position, size, shape
   - Detect relations: above, inside, adjacent

2. Rule inducer: observations → candidate SCM
   - For each example pair: what changed?
   - Abduct minimal rule explaining the change
   - Intersect rules across examples → verified rule

3. Our existing det-cf engine: execute rule on test
   - Already proven at 100%
   - No modification needed

4. Online update loop (the interactive part):
   - Take action → observe outcome
   - If outcome matches prediction: rule confirmed
   - If not: revise SCM (our Code-Graph mechanism)

## Why this beats LLMs structurally

LLM approach to ARC:
  "This looks like rotation, I've seen rotation patterns..."
  → Fails on novel rule combinations

SPE approach:
  "Object A is red and above Object B."
  "After transformation: Object A is blue."  
  "Rule: color(A) = f(position(A,B))"
  "Test: apply rule → predict"
  → Generalizes exactly because rule is explicit

## Timeline to ARC-AGI-3 attempt

Week 1-2: Grid encoder (object detection from pixel grid)
Week 3-4: Rule inducer (abduction from example pairs)  
Week 5-6: Integration + ARC-AGI-2 baseline (static)
Week 7-8: Interactive loop for ARC-AGI-3
Week 9+:  Optimize and score

## The pitch evolution

Before ARC-AGI-3:
  "We beat GPT-4 by +28.6% on causal reasoning"

After ARC-AGI-3 (if we score >10%):
  "We solved the benchmark frontier models score <1% on,
   using a system that runs on a consumer GPU"

That is a Series A conversation.
