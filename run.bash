declare -a args=(${@#})

main()
{
    initialize_arguments
    parse_arguments
    check_arguments
    
    docker_run 
}

initialize_arguments()
{
    container_name=""
    image_name=""
    cfg_dir=""
    poses_dir=""
    ckpt=""
}

parse_arguments()
{
    for ((i=0; i<${#args[@]}; i++)); do
        if [ "${args[$i]}" == "--container-name" ]; then
            container_name=${args[$(($i+1))]}
        fi
        if [ "${args[$i]}" == "--image-name" ]; then
            image_name=${args[$(($i+1))]}
        fi
        if [ "${args[$i]}" == "--cfg-dir" ]; then
            cfg_dir=${args[$(($i+1))]}
        fi
        if [ "${args[$i]}" == "--poses-dir" ]; then
            poses_dir=${args[$(($i+1))]}
        fi
        if [ "${args[$i]}" == "--ckpt" ]; then
            ckpt=${args[$(($i+1))]}
        fi
    done
}

check_arguments()
{
    if [ "$container_name" == "" ]; then
        echo "Error: container-name is not specified. (use --container-name)"
        echo "Exiting."
        exit
    fi
    if [ "$image_name" == "" ]; then
        echo "Error: image-name is not specified. (use --image-name)"
        echo "Exiting."
        exit
    fi
    if [ "$cfg_dir" == "" ]; then
        echo "Error: cfg_dir is not specified. (use --cfg-dir)"
        echo "Exiting."
        exit
    fi
    if [ "$poses_dir" == "" ]; then
        echo "Error: poses_dir is not specified. (use --poses-dir)"
        echo "Exiting."
        exit
    fi
    if [ "$ckpt" == "" ]; then
        echo "Error: ckpt is not specified. (use --ckpt)"
        echo "Exiting."
        exit
    fi
}

docker_run()
{
    XAUTH=/tmp/.docker.xauth

    echo "Preparing Xauthority data..."
    xauth_list=$(xauth nlist :0 | tail -n 1 | sed -e 's/^..../ffff/')
    if [ ! -f $XAUTH ]; then
        if [ ! -z "$xauth_list" ]; then
            echo $xauth_list | xauth -f $XAUTH nmerge -
        else
            touch $XAUTH
        fi
        chmod a+r $XAUTH
    fi

    echo "Done."
    echo ""
    echo "Verifying file contents:"

    # If not working, first do: sudo rm -rf /tmp/.docker.xauth
    # It still not working, try running the script as root.

    file $XAUTH
    echo "--> It should say \"X11 Xauthority data\"."
    echo ""
    echo "Permissions:"
    ls -FAlh $XAUTH
    echo ""
    echo "Running docker..."

    docker run --gpus all -it \
        --shm-size=4g \
        --env="DISPLAY=$DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --env="XAUTHORITY=$XAUTH" \
        --volume="$XAUTH:$XAUTH" \
        --net=host \
        --privileged \
        -v $poses_dir:/root/colmap_output \
        -v $ckpt:/root/weight.ckpt \
        -v $cfg_dir:/root/catkin_ws/src/Loc-NeRF/cfg \
        -v $PWD/src/nerfstudio:/root/catkin_ws/src/Loc-NeRF/src/nerfstudio \
        -v $PWD/src/loc_nerf/full_filter.py:/root/catkin_ws/src/Loc-NeRF/src/full_filter.py \
        -v $PWD/src/loc_nerf/nav_node.py:/root/catkin_ws/src/Loc-NeRF/src/nav_node.py \
        -v $PWD/src/loc_nerf/navigator_base.py:/root/catkin_ws/src/Loc-NeRF/src/navigator_base.py \
        -v $PWD/src/loc_nerf/particle_filter.py:/root/catkin_ws/src/Loc-NeRF/src/particle_filter.py \
        -v $PWD/src/loc_nerf/nerfacto_loader.py:/root/catkin_ws/src/Loc-NeRF/src/nerfacto_loader.py \
        -v $PWD/src/pose_regressor:/root/catkin_ws/src/Loc-NeRF/src/pose_regressor \
        -v $PWD/src/run_posenet.py:/root/catkin_ws/src/Loc-NeRF/src/run_posenet.py \
        -v $PWD/src/nerfacto_loader_2.py:/root/catkin_ws/src/Loc-NeRF/src/nerfacto_loader_2.py \
        -v $PWD/src/config_dfnet.txt:/root/catkin_ws/src/Loc-NeRF/src/config_dfnet.txt \
        --name $container_name \
        $image_name \
        bash

    echo "Done."
}

main 