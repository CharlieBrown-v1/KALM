# Knowledgeable Agents by Offline Reinforcement Learning from Large Language Model Rollouts

## Demonstration of imaginary rollouts

| Environments | Description | Demo     |
|  :----:  | :----:  | :----:  |
|ShadowHand Over| These environments involve two fixed-position hands. The hand which starts with the object must find a way to hand it over to the second hand. | <img src="assets/image_folder/0v2.gif" width="250"/>    |
|ShadowHandCatch Underarm|These environments again have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained region. | <img src="assets/image_folder/hand_catch_underarmv2.gif" align="middle" width="250"/>    |
|ShadowHandCatch Over2Underarm| This environment is made up of half ShadowHandCatchUnderarm and half ShadowHandCatchOverarm, the object needs to be thrown from the vertical hand to the palm-up hand | <img src="assets/image_folder/2v2.gif" align="middle" width="250"/>    |
|ShadowHandCatch Abreast| This environment is similar to ShadowHandCatchUnderarm, the difference is that the two hands are changed from relative to side-by-side posture. | <img src="assets/image_folder/1v2.gif" align="middle" width="250"/>    |
|ShadowHandCatch TwoCatchUnderarm| These environments involve coordination between the two hands so as to throw the two objects between hands (i.e. swapping them). | <img src="assets/image_folder/two_catchv2.gif" align="middle" width="250"/>    |
|ShadowHandLift Underarm | This environment requires grasping the pot handle with two hands and lifting the pot to the designated position  | <img src="assets/image_folder/3v2.gif" align="middle" width="250"/>    |
|ShadowHandDoor OpenInward | This environment requires the closed door to be opened, and the door can only be pulled inwards | <img src="assets/image_folder/door_open_inwardv2.gif" align="middle" width="250"/>    |
|ShadowHandDoor OpenOutward | This environment requires a closed door to be opened and the door can only be pushed outwards  | <img src="assets/image_folder/open_outwardv2.gif" align="middle" width="250"/>    |
|ShadowHandDoor CloseInward | This environment requires the open door to be closed, and the door is initially open inwards | <img src="assets/image_folder/close_inwardv2.gif" align="middle" width="250"/>    |
|ShadowHand BottleCap | This environment involves two hands and a bottle, we need to hold the bottle with one hand and open the bottle cap with the other hand  | <img src="assets/image_folder/bottle_capv2.gif" align="middle" width="250"/>    |
|ShadowHandPush Block | This environment requires both hands to touch the block and push it forward | <img src="assets/image_folder/push_block.gif" align="middle" width="250"/>    |
|ShadowHandOpen Scissors | This environment requires both hands to cooperate to open the scissors | <img src="assets/image_folder/scissors.gif" align="middle" width="250"/>    |
|ShadowHandOpen PenCap | This environment requires both hands to cooperate to open the pen cap  | <img src="assets/image_folder/pen.gif" align="middle" width="250"/>    |
|ShadowHandSwing Cup | This environment requires two hands to hold the cup handle and rotate it 90 degrees | <img src="assets/image_folder/swing_cup.gif" align="middle" width="250"/>    |
|ShadowHandTurn Botton | This environment requires both hands to press the button | <img src="assets/image_folder/switch.gif" align="middle" width="250"/>    |
|ShadowHandGrasp AndPlace | This environment has a bucket and an object, we need to put the object into the bucket  | <img src="assets/image_folder/g&p.gif" align="middle" width="250"/>    |