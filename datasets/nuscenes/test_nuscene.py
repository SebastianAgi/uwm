from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='/home/Sebastian/dataset/sets/nuscenes', verbose=True)
# nusc.list_scenes()
my_scene = nusc.scene[0]
# print(my_scene)
first_sample_token = my_scene['first_sample_token']
# nusc.render_sample(first_sample_token)
my_sample = nusc.get('sample', first_sample_token)
# print(my_sample)
# nusc.list_sample(my_sample['token'])
# print(my_sample['data'])
sensor = 'CAM_FRONT'
cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
print(cam_front_data)

nusc.render_sample_data(cam_front_data['token'])