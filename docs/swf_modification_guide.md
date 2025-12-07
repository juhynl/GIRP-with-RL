# Modifying GIRP.swf for a Reinforcement Learning Agent

This document focuses on modifying the ActionScript code of the game GIRP to allow a Reinforcement Learning (RL) agent to interact with and play the game.

## 1. Prerequisites: Get the Game File

Before you begin, you need the original girp.swf file.

Go to the official website: https://www.foddy.net/GIRP.html

Find the download links on the page.

Download the "windows build". This will provide you with the girp.swf file.

## 2. Tools

To decompile and recompile the SWF, you will likely need tools such as SWF Decompiler (e.g., JPEXS Free Flash Decompiler) to extract the ActionScript code.

## 3. Modify Codes

The following code must be added to the decompiled ActionScript files to expose the game state.

### Player.as
Add the following public methods inside the Player class.
These helper functions retrieve the current grip state (gripping or not) for each hand.

```as3
public function getLeftHandState() : Number
    {
        return gripArray["leftHand"];
    }

public function getRightHandState() : Number
    {
        return gripArray["rightHand"];
    }
```

### PlayState.as
Add the following import at the top of the file.

```as3
import flash.external.ExternalInterface;
```

Add these variables inside the PlayState class definition.

```as3
private const SPEED_SCALE:Number = 10
public var winResult:int = 0;
public var endTime:Number = 0;
public var hiScore:Number = 0;
```

The RL agent (Python) needs to call functions inside the Flash game. You must register these functions using `ExternalInterface`.

```as3
if(ExternalInterface.available)
{
   try
   {
      // Allows Python to call these functions
      ExternalInterface.addCallback("getStateForAgent", getStateForAgent);
      ExternalInterface.addCallback("setGoalUpper", setGoalUpper);
   }
   catch(error:Error)
   {
      FlxG.log("ExternalInterface Error: " + error.message);
   }
}
```

Add the `getStateForAgent()` method to the PlayState class. This function collects all relevant game data (player physics, hand states, bird, and toeholds) into a single Object for the RL agent.

```as3
public function getStateForAgent() : Object
{

    var state:Object = {};
    var refX:Number = 0;
    var refY:Number = 0;
    if(_player)
    {
        refX = _player._chest.GetPosition().x;
        refY = _player._chest.GetPosition().y;
        state.player = {
            "head":{
                "x":_player._head.GetPosition().x - refX,
                "y":_player._head.GetPosition().y - refY,
                "vx":_player._head.GetLinearVelocity().x,
                "vy":_player._head.GetLinearVelocity().y
            },
            "chest":{
                "x":0,
                "y":0,
                "vx":_player._chest.GetLinearVelocity().x,
                "vy":_player._chest.GetLinearVelocity().y
            },
            "chest_ref":{
                "x":refX,
                "y":refY,
                "vx":_player._chest.GetLinearVelocity().x,
                "vy":_player._chest.GetLinearVelocity().y
            },
            "leftHand":{
                "x":_player._leftHand.GetPosition().x - refX,
                "y":_player._leftHand.GetPosition().y - refY,
                "vx":_player._leftHand.GetLinearVelocity().x,
                "vy":_player._leftHand.GetLinearVelocity().y,
                "state":_player.getLeftHandState()
            },
            "rightHand":{
                "x":_player._rightHand.GetPosition().x - refX,
                "y":_player._rightHand.GetPosition().y - refY,
                "vx":_player._rightHand.GetLinearVelocity().x,
                "vy":_player._rightHand.GetLinearVelocity().y,
                "state":_player.getRightHandState()
            },
            "joints":{
                "leftShoulder":_player._leftShoulder.GetJointAngle(),
                "leftElbow":_player._leftElbow.GetJointAngle(),
                "rightShoulder":_player._rightShoulder.GetJointAngle(),
                "rightElbow":_player._rightElbow.GetJointAngle()
            }
        };
    }
    
    state.environment = {};

    state.environment.waterLevel = _player._head.GetPosition().y * ratio - (118 - _waterLevel);
    state.environment.worldPaused = _worldPaused;
    state.environment.time = _time;
    state.environment.winResult = winResult;
    state.environment.hiScore = hiScore;

    if(_bird)
    {
        state.environment.bird = {
            "x":_bird._obj.GetPosition().x - refX,
            "y":_bird._obj.GetPosition().y - refY,
            "landed":_bird._landed
        };
    }

    state.environment.toeholds = [];
    state.environment.toehold_indices = [];

    for each(var th in _toeholdsOnScreen)
    {
        if(th._letterVisible && th.state != 3)
        {
            state.environment.toeholds.push({
                "x":th._obj.GetPosition().x - refX,
                "y":th._obj.GetPosition().y - refY,
                "letter":th._letter,
                "state":th.state,
                "type":th._ringType
            });
            state.environment.toehold_indices.push(th._letter.charCodeAt(0) - "A".charCodeAt(0));
        }
    }

    return state;

}
```
Add the `setGoalHeight(targetY:Number)` function to move the goal ring closer and help the agent learn.
```as3
public function setGoalUpper(targetY:Number) : void
{
    for each(var th in _toeholdsOnScreen)
    {
        if(th._ringType == 2)
        {
            th._ringType = 1;
            th._sprite.play("main");
            th._sprite.frame = 0;
        }

        var dist:Number = targetY - th._obj.GetPosition().y;
        
        if(dist > 0)
        {
            th._ringType = 2;
            th._sprite.play("prize");
        }
    }
}
```



To train an RL agent efficiently, the game needs to run faster than real-time.
Wrap the `_world.Step()` in a loop based on `SPEED_SCALE`.

```as3
// Original Code: _world.Step(FlxG.elapsed, 10, 10);

var i:int = 0;
while(i < SPEED_SCALE)
{
   _world.Step(FlxG.elapsed, 12, 12); // Iterations increased to 12 for stability at high speed
   i++;
}
```

The agent needs to know immediately when the game ends. Modify the `update()`, `birdWins()`, and `playerWins()` functions to update the `winResult` variable.

```as3
override public function update() : void
{
    // ... existing code ...
    if(!_worldPaused)
    {
        if(_player._head.GetPosition().y * ratio > 118 - _waterLevel)
        {
            _player.dropDead();
            winResult = -1; // Update winResult on death
        }
    }
    // ... existing code ...
}

public function birdWins() : void
{
    winResult = -1; // Update winResult when the bird wins
}

public function playerWins() : void
{
    winResult = 1; // Update winResult when the player wins
}
```
