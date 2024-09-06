### Entropy
_States with many choices are often good_ **Entropy doesn't do that**
- Raise entropy coefficient 
- Use entropy in advantage estimation


### Value

- High coefficient, but clipped. _as low V high E led to "no score peak" training, try the other way around, but clip V to avoid catastrophic loss_

### Debugging

- Visualize _per action_ policy loss


### Random

- Remove value calc again
- Advantage batch normalization
