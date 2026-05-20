# Lay Summary

**Title:** Test-Time Graph Search for Goal-Conditioned Reinforcement Learning

Teaching an AI agent to perform a long sequence of actions is much harder than teaching it short ones: small mistakes accumulate, and the agent gets lost. Existing fixes add generative planners or extra neural networks, but they are heavy and require redoing the training pipeline. We asked whether an agent that already takes reliable short steps could handle long journeys without any retraining.

We built Test-Time Graph Search (TTGS), a lightweight tool that wraps around an already-trained agent. It treats observations from the training data as waypoints on a map and connects them using the agent's own sense of how close two situations are. A shortest-path search picks a chain of nearby waypoints that guides the agent one short step at a time. Because the agent's sense of distance is trustworthy for nearby places but noisy for far ones, we bias the search toward paths made of many small hops rather than a few long, uncertain ones.

On a standard benchmark, TTGS lifted success rates on the hardest navigation tasks from near zero to over 90%, while adding less than a second of computation. This lets practitioners unlock long-horizon performance from agents they have already trained, without extra data or training.
