# Sim Scripting Intro

In the sim, objects are interacted with via `handles`, which are ints that correspond to a particular object 

To get an object handle from a name, do `sim.getObjectHandle('ObjectName')` 

# Useful Sim Methods
- `sim.getObjectPosition(handle, relToHandle)`, where relToHandle is the handle of the frame you want the position in. `-1` for the global frame 
- `sim.getObjectMatrix(handle, relToHandle)` for object's rotation matrix
- `sim.getObjectQuaternion(handle, relToHandle)` for object's orientation as quaternion
- TODO: whatever I did to walk the object tree and make them semi-transparent

# Outputting transforms to ROS
The sim can put transforms directly into the ROS `/tf` topic! 

1. Copypaste this function into the script of the component you want the transform for:
    ```
    -- copypasted from https://forum.coppeliarobotics.com/viewtopic.php?t=6198
    function getTransformStamped(objHandle,name,relTo,relToName)
        t=sim.getSystemTime()
        p=sim.getObjectPosition(objHandle,relTo)
        o=sim.getObjectQuaternion(objHandle,relTo)

        return {
            header={
                stamp=t,
                frame_id=relToName
            },
            child_frame_id=name,
            transform={
                -- ROS has definition x=front y=side z=up
                translation={x=p[1],y=p[2],z=p[3]},--V-rep
                rotation={x=o[1],y=o[2],z=o[3],w=o[4]}--v-rep
            }
        }
    end
    ```
    `name` is the name of the child frame, and `relToName` is the name of the parent frame. This function returns a table that is equivalent to the ROS `/tf` message with the same fields. 
2. Send the transform
    ```
    tf = getTransformStamped(J1, 'J1_PSM1', -1, 'simworld')
    simROS.sendTransform(tf)
    ```