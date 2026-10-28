/* REQ: whenever the train is in the crossing, the gate must be down (part 1/2) */
A[] (train.Cross imply gate.Down)
/* REQ: whenever the train is in the crossing, the gate must be down (part 2/2) */
E<> train.Cross

/* REQ: The train should never stay in the crossing for more than 5 time units. (part 1/2) */
A[] (train.Cross imply train.x <= 5)
/* REQ: The train should never stay in the crossing for more than 5 time units. (part 2/2) */
E<> train.Cross
