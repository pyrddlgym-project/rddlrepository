# A Guide for the BLXed

The goal of this document is to describe the BLX traffic model, as well as some of the subtle points of
its implementation in the RDDL language. The title is a play on "A Guide for the Perplexed",
a work of philosophy by Maimonides. The author apologizes for the pun, but it seemed too good to pass up :)

The rest of the document is structured as follows. The first section will describe the BLX model in general terms,
discuss the issue of how vehicle flow propagation is implemented in RDDL, and discuss linear blending. The
second section will discuss the difference between so-called "Simple" and "Complex" phasing schemes.
The final section will describe how to use the instance file generator to create a grid network and run
the instance as a RDDLEnv using pyRDDLGym.

## The van den Berg - Lin - Xi (BLX) model
A link is a piece of road connecting two intersections or connecting an intersection to the boundary of
the traffic network. If we think of the traffic network as a directed graph, "link" is another name for
a directed edge. Each link may have several incoming turns from other links, and several outgoing turns
to other links. Each turn has some number of lanes (for simplicity we assume that the number of lanes
does not change throughout the link).

The van den Berg - Lin - Xi (BLX) model is a traffic flow model that strikes a good balance between simplicity
and detail. The model operates in discrete time-steps of equal duration (although in principle the duration
may be made to vary, as was done in the similar Queue Transmission Model (QTM)). For each outgoing turn on a
link, the model keeps track of:
 - The current queue length (number of stopped vehicles at the downstream end of the link)
 - The flows along the link

Incoming flows are assumed to propagate at some constant speed along the link (the value of the speed may depend on the link),
until joining the end of a queue. The outgoing flow of a turn that has a green light is determined as the *minimum*
of the following three terms:
 - Queue length + Current incoming flow
 - Saturation flow rate (parameter of a turn)
 - Storage capacity of the downstream (target) link (parameter of a link)
In this way, BLX models three different traffic modalities:
 - Undersaturated flow (there are few enough vehicles in the queue and currently arriving that they can be totally cleared)
 - Saturated flow (as many vehicles as possible are cleared in the time-step)
 - Oversaturated flow (saturated flow cannot be reached because of limits on downstream capacity)

A reference for the BLX model is

 > S. Lin, B. De Schutter, Y. Xi, and J. Hellendoorn, "A simplified macroscopic urban
   traffic network model for model-based predictive control," Proceedings of the 12th
   IFAC Symposium on Transportation Systems, Redondo Beach, California, pp. 286--291,
   Sept. 2009

the similar Queue Transmission Model appeared in

 > Guilliard, I., Sanner, S., Trevizan, F. W., & Williams, B. C. "Nonhomogeneous
   time mixed integer linear programming formulation for traffic signal control,"
   Transportation Research Record, pp. 128-138 2595(1), 2016

### Description of the model
### Vehicle flow propagation in RDDL (Why is time encoded as an object?)
### Linear blending of incoming flows

## Simple and Complex phasing structures
### Simple phasing
### Complex phasing

## Start-up guide
### Generating a domain instance
### Running and interfacing with an instance as a RDDLEnv
### Viewing in the visualizer
