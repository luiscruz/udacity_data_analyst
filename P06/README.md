## Summary
Ranking of reasons to uninstall a mobile Application
Ling, Soo et al. conducted a study about mobile app users behaviors. They surveyed 10,208 people from more than 15 countries on their mobile app usage behavior.
Using their dataset, I have analyzed _what makes a user stop using an app?_.
Results were split between Android and iOS platforms and displayed using a bar chart.

## Design

I've decided to use a horizontal bar chart using Dimple.js. Although this is a very basic chart, that is exactly the point. Adding more features to the visualization would require extra effort from readers.

The color was used to distinguish the platform.
Length was used to code the portion of users. Since a participant could give more than one answer, the weight of his/her answer was divided by the number of answers he/she gave.
The position was used for the answer selected by the participant.

## Feedback

#### Rui ([ruimaranhao](https://discussions.udacity.com/users/ruimaranhao/activity))

> This visualization is interesting. I wonder if you could show the same message for different demographics?

#### Andrew ([CtrlAltDel](https://discussions.udacity.com/users/CtrlAltDel/activity))

> @luismirandacruz,

> Very clean visualization. I like how the intro started off with the key literature reference. I think most people would be able to pick some key findings from this visualization.One overall finding that I find interesting is that Android and Apple users seems to agree on what influences them to drop an app, so for the most part whether or not to keep or drop an app is largely platform independent. What might be interesting is to determine where reasoning is not platform independent. This could make for an important business insight.
> Per the suggestion offered by @ruimaranhao showing results per some other factor (country for example) would be very informative.

> The one big recommendation I offer is to consider changing the color palette to a color blind friendly palette. Blue green color blindness is very common, especially among men, and this color scheme could be problematic. Whenever I get visualization feedback I try to include someone who is colorblind, just to make sure that my color choices are not impacting that segment of the population.

Note that some recommended palettes can be found in the course materials.

### Improvements

Unfortunately, the dataset made public does not provide demographics or any other personal information about participants.
Thus, some of the suggested improvements could not be implemented.

The color palette used was reviewed. Colorblind readers should not have a problem reading the visualization and less bright colors were used.


## Resources

- https://soolinglim.wordpress.com/datasets/
- Soo Ling Lim, Peter J. Bentley, Natalie Kanakam, Fuyuki Ishikawa, and Shinichi Honiden (2015). Investigating Country Differences in Mobile App User Behavior and Challenges for Software Engineering. IEEE Transactions on Software Engineering (TSE), vol 41 issue 1, pp 40-64.