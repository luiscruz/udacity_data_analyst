## Summary
Ranking of reasons to uninstall a mobile Application
Ling, Soo et al. conducted a study about mobile app users behaviors. They surveyed 10,208 people from more than 15 countries on their mobile app usage behavior.
Using their dataset, I have analyzed _what makes a user stop using an app?_.
Results were split between Android and iOS platforms and displayed using a bar chart.

## Design

I've decided to use a horizontal bar chart using Dimple.js. Although this is a very basic chart, that is exactly the point. Adding more features to the visualization would require extra effort from readers.

The color was used to distinguish the platform.
Length was used to code the portion of users. 
The position was used for the answer selected by the participant.

The final version of the visualization can be reached [here](https://rawgit.com/luiscruz/udacity_data_analyst/master/P06/index.html).

## Findings
- Android and iOS users agree reasons to stop using an app
- Slow performance causes more Android users to stop using an app than iPhone users
- The visualization suggests that privacy is not a big concern among users of both platforms (4%).

## Feedback

### Feedback at [Udacity Discussion Forum](https://discussions.udacity.com/t/p06-feedback-request-on-what-makes-you-stop-using-an-app/190034)

#### Rui ([ruimaranhao](https://discussions.udacity.com/users/ruimaranhao/activity))

> This visualization is interesting. I wonder if you could show the same message for different demographics?

#### Andrew ([CtrlAltDel](https://discussions.udacity.com/users/CtrlAltDel/activity))

> @luismirandacruz,

> Very clean visualization. I like how the intro started off with the key literature reference. I think most people would be able to pick some key findings from this visualization.One overall finding that I find interesting is that Android and Apple users seems to agree on what influences them to drop an app, so for the most part whether or not to keep or drop an app is largely platform independent. What might be interesting is to determine where reasoning is not platform independent. This could make for an important business insight.
> Per the suggestion offered by @ruimaranhao showing results per some other factor (country for example) would be very informative.

> The one big recommendation I offer is to consider changing the color palette to a color blind friendly palette. Blue green color blindness is very common, especially among men, and this color scheme could be problematic. Whenever I get visualization feedback I try to include someone who is colorblind, just to make sure that my color choices are not impacting that segment of the population.

> Note that some recommended palettes can be found in the course materials.

### Feedback at [Google+](https://plus.google.com/100680226865763721677/posts/gMBhUaoW9cx)
This feedback was given upon the second version of the visualization. In this version, colors had already been reviewed.

#### [Eilan Ou](https://plus.google.com/113309008589435841729)

> - What is the first thing you notice in the visualization? iOS and Android users have similar responses
> - Is the visualization appealing? Is it simple? It's simple. Contrasting but softened color choices are good. 
> - What is the main message? Many apps are not needed for users.
> - What do you think would make the message clearer? Groups response categories
> - How would you improve the visualization? My instinct is that people are more familiar with percentage than portion. And personally I want to compare iOS/Android more easily.

> Hope this helps :-)


### Improvements

Unfortunately, the dataset made public by the authors does not provide demographics or any other personal information about participants.
Thus, some of the suggested improvements could not be implemented.

The color palette used was reviewed. Colorblind readers should not have a problem reading the visualization. In addition, less bright colors were used to make sure the visualization is not uncomfortable to the eyes of the reader.

According to Eilan, having values in terms of portion is not intuitive.
I've changed the visualization to show percentage, since readers are more familiar with it.


## Resources

- https://soolinglim.wordpress.com/datasets/
- Soo Ling Lim, Peter J. Bentley, Natalie Kanakam, Fuyuki Ishikawa, and Shinichi Honiden (2015). Investigating Country Differences in Mobile App User Behavior and Challenges for Software Engineering. IEEE Transactions on Software Engineering (TSE), vol 41 issue 1, pp 40-64.