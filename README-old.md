## TODO

* [ ] Landmakr video clip handling
  
    * take snapshot --> process it --> collect information
* [ ] Deploy operator
* [ ] start querying --> train operator based on collected landmark frames --> Deploy operator --> ranked video frames --> sending imgs -> YOLO again --> ok --> ask camera send video clip (5 sec)

* Deploy operator & ranked video frames

* yolo_service api: searching for certain object(s):
  
* search_objects(image, [object_names]) -> bbox
  
* contraoller api: receive video and store video on disk

            # FIXME step 1. find camera information
            # element exist
            # step 2. creating video record, requesting video
            # step 3. implementing new api interface to receive video
            #  from camera
            # step 4. process video