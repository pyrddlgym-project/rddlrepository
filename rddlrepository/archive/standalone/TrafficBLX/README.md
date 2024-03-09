# A Guide for the BLXed

The goal of this document is to describe the BLX traffic model, as well as some of the subtle points of
its implementation in the RDDL language.

The document is structured as follows. The first section will describe the BLX model in general terms,
discuss the issue of how vehicle flow propagation is implemented in RDDL, and discuss linear blending. The
second section will discuss the difference between so-called "Simple" and "Complex" phasing schemes.
The final section will describe how to use the instance file generator to create a grid network and run
the instance as a RDDLEnv using pyRDDLGym.

(What's with the title? The title is a play on "A Guide for the Perplexed", a work of philosophy by Maimonides.
The author apologizes for the pun, but it seemed too good to pass up :))


## The van den Berg - Lin - Xi (BLX) model
A link is a piece of road connecting two intersections or connecting an intersection to the boundary of
the traffic network. If we think of the traffic network as a directed graph, "link" is another name for
a directed edge. Each link may have several incoming turns from other links, and several outgoing turns
to other links. Each turn has some number of incoming lanes, which affects its saturation flow rate
(for simplicity we assume that the number of lanes does not change throughout the link). The number of lanes
on the link is equal to the sum of the number of lanes over all of its outgoing turns.

The van den Berg - Lin - Xi (BLX) model is a traffic flow model that strikes a good balance between simplicity
and detail. The model operates in discrete time-steps of equal duration (although in principle the duration
may be made to vary, as was done in the similar Queue Transmission Model (QTM)). For each outgoing turn on a
link, the model keeps track of:
 - The current queue length (number of stopped vehicles at the downstream end of the link)
 - The flows along the link

Incoming flows are split into different outgoing turns by multiplying the flow by the turning probability of the turn.
The flows that are split according to their outgoing turn are then assumed to propagate at some constant speed along 
the link (the value of the speed may depend on the link), until joining the end of a queue for that turn.
The outgoing flow of a turn that has a green light is determined as the *minimum* of the following three terms:
 - The sum queue length + current incoming flow
 - Saturation flow rate (parameter of a turn)
 - Remaining storage of the downstream (target) link (the maximal storage capacity is a parameter of a link)

In this way, BLX models three different traffic modalities. Respectively, they are:
 - Undersaturated flow (there are few enough vehicles in the queue and currently arriving that they can be totally cleared)
 - Saturated flow (as many vehicles as possible are cleared in the time-step)
 - Oversaturated flow (saturated flow cannot be reached because of limits on downstream capacity)

More detailed information may be found by inspecting the domain RDDL file, or referring to the paper

 > S. Lin, B. De Schutter, Y. Xi, and J. Hellendoorn, "A simplified macroscopic urban
   traffic network model for model-based predictive control," Proceedings of the 12th
   IFAC Symposium on Transportation Systems, Redondo Beach, California, pp. 286--291,
   Sept. 2009

the similar Queue Transmission Model appeared in

 > Guilliard, I., Sanner, S., Trevizan, F. W., & Williams, B. C. "Nonhomogeneous
   time mixed integer linear programming formulation for traffic signal control,"
   Transportation Research Record, pp. 128-138 2595(1), 2016

We now describe two subtleties in the RDDL implementation of the BLX model.

### Vehicle flow propagation in RDDL (Why is time encoded as an object?)
A special feature of the BLX model that is a bit tricky to deal with from the MDP perspective is that the incoming flows
join the end of a queue after a time offset (the time it takes the flow to propagate from entrance until reaching the queue).
It becomes necessary to keep information from previous time-steps as part of the state.

### Linear blending of incoming flows

## Simple and Complex phasing structures
### Simple phasing
### Complex phasing

## Start-up guide
### Generating a domain instance
### Running and interfacing with an instance as a RDDLEnv
### Viewing in the visualizer
