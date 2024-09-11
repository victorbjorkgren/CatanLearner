### Value Learning
_We have seen great increases with better stabiltiy in gradients_
- Lower grad clip _no effect at 0.1_ **try 0.07**
- Larger net **try 64 hidden nodes**

### Entropy
_States with many choices are often good_ **Entropy doesn't do that**
- Give bonus for large action spaces 


- Raise entropy coefficient 
- Include entropy in advantage estimation


### Debugging

- Visualize _per action_ policy loss


### Random

- Remove value calc again
- Advantage batch normalization
