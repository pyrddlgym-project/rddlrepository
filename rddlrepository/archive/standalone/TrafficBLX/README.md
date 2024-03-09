# A Guide for the BLXed

The goal of this document is to describe the BLX traffic model, as well as some of the important and 
subtle details of its implementation in the RDDL language.

The document is structured as follows. The first section will describe the BLX model in general terms,
discuss the issue of how vehicle flow propagation is implemented in RDDL, and discuss linear blending of
incoming flows. The second section will discuss the difference between so-called "Simple" and "Complex"
phasing schemes. The final section will describe how to use the instance file generator to create a grid
network and run the instance as a RDDLEnv using pyRDDLGym.

(What's with the title? The title is a play on "A Guide for the Perplexed", a work of philosophy by Maimonides.
The author apologizes for the pun, but it seemed too good to pass up :))


## The van den Berg - Lin - Xi (BLX) model
A link is a piece of road connecting two intersections or connecting an intersection to the boundary of
the traffic network. If we think of the traffic network as a directed graph, "link" is another name for
a directed edge. Each link may have several incoming turns from other links, and several outgoing turns
to other links. Each turn has some number of incoming lanes, which affects its saturation flow rate, i.e.
the highest rate at which the turn can drain traffic (for simplicity we assume that the number of lanes does
not change throughout the link). The number of lanes on the link is equal to the sum of the number of lanes 
over all of its outgoing turns.

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
   Transportation Research Record, pp. 128--138 2595(1), 2016

We now describe two subtleties in the RDDL implementation of the BLX model.

### Vehicle flow propagation in RDDL (Why is time encoded as an object?)
A special feature of the BLX model that is a bit tricky to deal with from the MDP perspective is that the incoming flows
join the end of a queue after a time offset (the time it takes the flow to propagate from entrance until reaching the queue).
It becomes necessary to keep information from previous time-steps as part of the state.

For example, let us imagine that the link is 100 m long, and the propagation speed is 20 m/s. In addition, let the model operate with a time-step of 1 s.
First, imagine that the queues are empty. Then the vehicle flows that are entering the link at the current time-step will be arriving at the downstream end of the link in
100/20 = 5 s (time-steps). For scheduling the traffic light phases, it is important to know the full picture of the incoming flows,
that is, how many vehicles will be arriving in 0, 1, 2, 3, 4 seconds.

We can encode all of this information into an array-like object
``` flow-on-link(time) ```
where ``time`` is a RDDL object that has instances t0, t1, t2, t3, and t4.

If the queues are not empty, we would like to find the time that the incoming vehicle flows take to reach the
*upstream end* of the queue. This time will necessarily be <= 5 seconds. Therefore, with non-empty queues we
can continue using the same ``flow-on-link(time)`` array-like object.

Although RDDL does not provide a native array-element-access mechanism, we can mimic this as follows. Each
time object has a ``TIME-VAL(time)`` non-fluent, which acts as the array index. Then, if ``tau`` denotes
the propagation time to the end of queue (it could be 3 seconds, for example) and ``flow-into-link`` denotes
the incoming vehicle flows, we can add the new flows as

``` flow-on-link'(?t) = (TIME-VAL(?t) == tau) * flow-into-link; ```

in addition, to propagate the flows along the link, we need the concept of a sequential order
(or ordinal structure) on the time objects. We implement this using the boolean non-fluent ``NEXT(?ta,?tb)``,
where for example ``NEXT(t3,t2)`` is true and ``NEXT(t4,t2)`` is false. Using this concept, we can propagate
flows as

``` flow-on-link'(?t) = (sum_{?tb : time} [ NEXT(?t,?tb) * flow-on-link(?tb) ]); ```

Putting the incoming and propagated flows together, we obtain the update rule

``` flow-on-link'(?t) = (TIME-VAL(?t) == tau) * flow-into-link + (sum_{?tb : time} [ NEXT(?t,?tb) * flow-on-link(?tb) ]); ```

### Linear blending of incoming flows
If we compare the update rule for ``flow-on-link(?t)`` described in the previous subsection with the update rule used in the
RDDL domain files, we see that there is an additional detail that is still missing. This is linear blending of incoming flows.

Imagine that we are getting a 10 vehicle inflow into our link, and that the estimated propagation time to the end
of the current queue is equal to 8.3 time-steps. Should we round up or round down 8.3 to find the time-index where
the inflow is inserted into the ``flow-on-link(?t)`` "array"? Because at different times of the simulation the
estimated propagation time could take values 8.01, 8.49, 8.51, or 8.99 (for example), the floor, round, and ceiling
operations can all be fairly inaccurate.

Therefore, BLX instead *linearly blends* the incoming flows into two adjacent entries of the ``flow-on-link(?t)``
array. Some of the inflow goes into the 8-second-offset and some of the inflow goes into the 9-second offset.
The split is done linearly. In the example, because 8.3 is closer to 8 than to 9, more of the inflow should
go into the 8-second-offset. (1-0.3)*10 goes into 8-second offset and 0.3*10 goes into 9-second offset.

More generally, we write the propagation estimate as ``tau + gamma``, where ``tau`` is an integer and ``gamma``
is a real number in the interval [0, 1). Then the incoming inflow is linearly blended as

``` flow-on-link'(?t) =  (TIME-VAL(?t) == tau) * (1-gamma) * flow-into-link
                       + (TIME-VAL(?t) == tau+1) * gamma   * flow-into-link; ```

We have now explained all of the elements of the ``flow-on-link`` update rule.

## Simple and Complex phasing structures
### Simple phasing
### Complex phasing

## Start-up guide
### Generating a domain instance
### Running and interfacing with an instance as a RDDLEnv
### Viewing in the visualizer
