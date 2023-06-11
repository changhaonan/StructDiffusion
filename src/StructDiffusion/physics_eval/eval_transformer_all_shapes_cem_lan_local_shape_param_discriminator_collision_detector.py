import torch
import numpy as np
import os
import argparse
import pytorch3d.transforms as tra3d
import json
import argparse
from omegaconf import OmegaConf

# physics eval
from StructDiffusion.utils.physics_eval import switch_stdout, visualize_batch_pcs, convert_bool, save_dict_to_h5, move_pc_and_create_scene_new, move_pc, fit_gaussians, sample_gaussians
from rearrangement_gym.semantic_rearrangement.physics_verification_dinner import verify_datum_in_simulation

# inference
from StructDiffusion.evaluation.infer_prior_continuous_out_encoder_decoder_struct_pct_6d_dropout_all_objects_all_shapes import PriorInference

# discriminators
from StructDiffusion.evaluation.infer_collision import CollisionInference
from StructDiffusion.evaluation.infer_discriminator import DiscriminatorInference



def evaluate(random_seed, structure_type,
            generator_model_dir, data_split, data_root,
            discriminator_model_dir=None, discriminator_model=None, discriminator_cfg=None, discriminator_tokenizer=None,
            collision_model_dir=None,
            collision_score_weight=0.5, discriminator_score_weight=0.5,
            ce_num_iteration=2, ce_num_samples=50, ce_num_elite=15, ce_num_best_so_far=1,
             discriminator_inference_batch_size=64,
             assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
            object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_large",
            redirect_stdout=False, shuffle=False, summary_writer=None, max_num_eval=10, visualize=False,
            override_data_dirs=None, override_index_dirs=None, physics_eval_early_stop=True, **kwargs):

    assert 0 <= collision_score_weight <= 1
    assert 0 <= discriminator_score_weight <= 1

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

    save_dir = os.path.join(data_root, data_split)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    OmegaConf.save(cfg, os.path.join(save_dir, "experiment_config.yaml"))

    if redirect_stdout:
        stdout_filename = os.path.join(data_root, "{}_log.txt".format(data_split))
    else:
        stdout_filename = None
    switch_stdout(stdout_filename)

    prior_inference = PriorInference(generator_model_dir, data_split=data_split,
                                     override_data_dirs=override_data_dirs,
                                     override_index_dirs=override_index_dirs)
    prior_dataset = prior_inference.dataset

    if discriminator_score_weight > 0:
        if discriminator_model_dir is not None:
            discriminator_inference = DiscriminatorInference(discriminator_model_dir)
            discriminator_model = discriminator_inference.model
            discriminator_cfg = discriminator_inference.cfg
            discriminator_tokenizer = discriminator_inference.dataset.tokenizer
        else:
            assert discriminator_model is not None
            assert discriminator_cfg is not None
            assert discriminator_tokenizer is not None
        discriminator_model.eval()
        discriminator_num_scene_pts = discriminator_cfg.dataset.num_scene_pts
        discriminator_normalize_pc = discriminator_cfg.dataset.normalize_pc
    else:
        discriminator_num_scene_pts = None
        discriminator_normalize_pc = False

    if collision_score_weight > 0:
        collision_inference = CollisionInference(collision_model_dir, empty_dataset=True)
        collision_model = collision_inference.model
        collision_cfg = collision_inference.cfg
        collision_model.eval()
        collision_num_pair_pc_pts = collision_cfg.dataset.num_scene_pts
        collision_normalize_pc = collision_cfg.dataset.normalize_pc
    else:
        collision_num_pair_pc_pts = None
        collision_normalize_pc = False

    device = prior_inference.cfg.device
    print("device", device)

    # params
    S = ce_num_samples
    B = discriminator_inference_batch_size

    if shuffle:
        prior_dataset_idxs = np.random.permutation(len(prior_dataset))
    else:
        prior_dataset_idxs = list(range(len(prior_dataset)))

    all_eval_idxs = []
    success_eval_idxs = []
    # debug
    all_scores_average = []
    for idx, data_idx in enumerate(prior_dataset_idxs):
        print("\n" + "*" * 50)

        ####################################################
        # sample S predictions

        sample_raw_data = prior_dataset.get_raw_data(data_idx, inference_mode=True)
        sample_file_id = os.path.splitext(os.path.basename(sample_raw_data["filename"]))[0]
        print(sample_file_id)
        samples_data = [prior_dataset.convert_to_tensors(sample_raw_data, prior_dataset.tokenizer)] * S


        ####################################################
        # sample S predictions

        # only running the following line once is incorrect because transformer is autoregressive
        # struct_preds, obj_preds = prior_inference.limited_batch_inference(samples_data, verbose=False, convert_to_tensors=False, return_numpy=False)

        # important: iterative sampling
        beam_data = [sample_raw_data] * S
        # first predict structure pose
        beam_goal_struct_pose, target_object_preds = prior_inference.limited_batch_inference(beam_data)
        for b in range(S):
            datum = beam_data[b]
            datum["struct_x_inputs"] = [beam_goal_struct_pose[b][0]]
            datum["struct_y_inputs"] = [beam_goal_struct_pose[b][1]]
            datum["struct_z_inputs"] = [beam_goal_struct_pose[b][2]]
            datum["struct_theta_inputs"] = [beam_goal_struct_pose[b][3:]]

        # then iteratively predict pose of each object
        beam_goal_obj_poses = []
        for obj_idx in range(prior_dataset.max_num_objects):
            struct_preds, target_object_preds = prior_inference.limited_batch_inference(beam_data)
            beam_goal_obj_poses.append(target_object_preds[:, obj_idx])
            for b in range(S):
                datum = beam_data[b]
                datum["obj_x_inputs"][obj_idx] = target_object_preds[b][obj_idx][0]
                datum["obj_y_inputs"][obj_idx] = target_object_preds[b][obj_idx][1]
                datum["obj_z_inputs"][obj_idx] = target_object_preds[b][obj_idx][2]
                datum["obj_theta_inputs"][obj_idx] = target_object_preds[b][obj_idx][3:]
        # concat in the object dim
        beam_goal_obj_poses = np.stack(beam_goal_obj_poses, axis=0)
        # swap axis
        beam_goal_obj_poses = np.swapaxes(beam_goal_obj_poses, 1, 0)  # batch size, number of target objects, pose dim

        struct_preds = torch.FloatTensor(beam_goal_struct_pose)
        obj_preds = torch.FloatTensor(beam_goal_obj_poses)

        # ToDo: figure out how to deal with struct preds, maybe taking an average
        struct_preds = struct_preds.mean(dim=0)
        struct_preds = struct_preds.repeat(S, 1)

        print(struct_preds.shape)
        print(obj_preds.shape)

        # struct_preds: S, 12
        # obj_preds: S, N, 12
        ####################################################
        # obj_xyzs: N, P, 3
        # obj_params: S, N, 6
        # struct_pose: S x N, 4, 4
        # current_pc_pose: S x N, 4, 4
        # target_object_inds: 1, N

        ####################################################
        # only keep one copy
        print("sentence", sample_raw_data["sentence"])

        # prepare for discriminator
        if discriminator_score_weight > 0:
            raw_sentence_discriminator = [sample_raw_data["sentence"][si] for si in [0, 4]]
            raw_sentence_pad_mask_discriminator = [sample_raw_data["sentence_pad_mask"][si] for si in [0, 4]]
            raw_position_index_discriminator = list(range(discriminator_cfg.dataset.max_num_shape_parameters))
            print("raw_sentence_discriminator", raw_sentence_discriminator)
            print("raw_sentence_pad_mask_discriminator", raw_sentence_pad_mask_discriminator)
            print("raw_position_index_discriminator", raw_position_index_discriminator)

        ####################################################
        # only keep one copy

        # N, P, 3
        obj_xyzs = samples_data[0]["xyzs"].to(device)
        print("obj_xyzs shape", obj_xyzs.shape)

        # 1, N
        # object_pad_mask: padding location has 1
        object_pad_mask = samples_data[0]["object_pad_mask"].to(device).unsqueeze(0)
        target_object_inds = 1 - object_pad_mask
        print("target_object_inds shape", target_object_inds.shape)
        print("target_object_inds", target_object_inds)

        N, P, _ = obj_xyzs.shape
        print("B, N, P: {}, {}, {}".format(B, N, P))
        if visualize:
            visualize_batch_pcs(obj_xyzs, 1, N, P)

        ####################################################
        # S, N, ...

        # Important: we don't optimize structure
        struct_pose = torch.eye(4).repeat(S, 1, 1).to(device)  # S, 4, 4
        struct_pose[:, :3, :3] = struct_preds[:, 3:].reshape(-1, 3, 3)
        struct_pose[:, :3, 3] = struct_preds[:, :3]
        struct_pose = struct_pose.repeat_interleave(N, dim=0)  # S x N, 4, 4

        current_pc_pose = torch.eye(4).repeat(N, 1, 1).to(device)  # N, 4, 4
        current_pc_pose[:, :3, 3] = torch.mean(obj_xyzs, dim=1)  # N, 4, 4
        current_pc_pose = current_pc_pose.repeat(S, 1, 1)  # S x N, 4, 4

        obj_params = torch.zeros((S, N, 6)).to(device)
        obj_preds = obj_preds.reshape(S, N, -1)  # S, N, 12
        obj_params[:, :, :3] = obj_preds[:, :, :3]
        obj_params[:, :, 3:] = tra3d.matrix_to_euler_angles(obj_preds[:, :, 3:].reshape(S, N, 3, 3), "XYZ")

        new_obj_xyzs_before_cem, goal_pc_pose_before_cem = move_pc(obj_xyzs, obj_params, struct_pose, current_pc_pose, device)

        if visualize:
            visualize_batch_pcs(new_obj_xyzs_before_cem, S, N, P, limit_B=5)

        ####################################################
        # CEM

        # intialize
        mus = None
        sigmas = None
        elite_scores = None

        # important: keep the best sample found in the cem process
        best_obj_params_so_far = [None] * ce_num_best_so_far
        best_score_so_far = [0] * ce_num_best_so_far

        # debug
        scores_average = []

        for ce_iter in range(ce_num_iteration):
            print("cross entropy optimization iteration:", ce_iter)

            # sampled_obj_params: S, N, 6
            if ce_iter == 0:
                sampled_obj_params = obj_params
            else:
                # sample_gaussians return [sample_size, number of individual gaussians]
                sampled_obj_params = sample_gaussians(mus, sigmas, sample_size=S).reshape(S, N, 6)

            # evaluate in batches
            scores = torch.zeros(S).to(device)
            no_intersection_scores = torch.zeros(S).to(device)  # the higher the better
            num_batches = int(S / B)
            if S % B != 0:
                num_batches += 1
            for b in range(num_batches):
                if b + 1 == num_batches:
                    cur_batch_idxs_start = b * B
                    cur_batch_idxs_end = S
                else:
                    cur_batch_idxs_start = b * B
                    cur_batch_idxs_end = (b + 1) * B
                cur_batch_size = cur_batch_idxs_end - cur_batch_idxs_start

                # print("current batch idxs start", cur_batch_idxs_start)
                # print("current batch idxs end", cur_batch_idxs_end)
                # print("size of the current batch", cur_batch_size)

                batch_obj_params = sampled_obj_params[cur_batch_idxs_start: cur_batch_idxs_end]
                batch_struct_pose = struct_pose[cur_batch_idxs_start * N: cur_batch_idxs_end * N]
                batch_current_pc_pose = current_pc_pose[cur_batch_idxs_start * N:cur_batch_idxs_end * N]

                new_obj_xyzs, _, subsampled_scene_xyz, _, obj_pair_xyzs = \
                    move_pc_and_create_scene_new(obj_xyzs, batch_obj_params, batch_struct_pose, batch_current_pc_pose,
                                                 target_object_inds, device,
                                                 return_scene_pts=discriminator_score_weight > 0,
                                                 return_scene_pts_and_pc_idxs=False,
                                                 num_scene_pts=discriminator_num_scene_pts,
                                                 normalize_pc=discriminator_normalize_pc,
                                                 return_pair_pc=collision_score_weight > 0,
                                                 num_pair_pc_pts=collision_num_pair_pc_pts,
                                                 normalize_pair_pc=collision_normalize_pc)

                # # debug:
                # print(subsampled_scene_xyz.shape)
                # print(subsampled_scene_xyz[0])
                # trimesh.PointCloud(subsampled_scene_xyz[0, :, :3].cpu().numpy()).show()

                #######################################
                # predict whether there are pairwise collisions
                if collision_score_weight > 0:
                    with torch.no_grad():
                        _, num_comb, num_pair_pc_pts, _ = obj_pair_xyzs.shape
                        # obj_pair_xyzs = obj_pair_xyzs.reshape(cur_batch_size * num_comb, num_pair_pc_pts, -1)
                        collision_logits = collision_model.forward(
                            obj_pair_xyzs.reshape(cur_batch_size * num_comb, num_pair_pc_pts, -1))
                        collision_scores = collision_model.convert_logits(collision_logits)["is_circle"].reshape(
                            cur_batch_size, num_comb)  # cur_batch_size, num_comb

                        # debug
                        # for bi, this_obj_pair_xyzs in enumerate(obj_pair_xyzs):
                        #     print("batch id", bi)
                        #     for pi, obj_pair_xyz in enumerate(this_obj_pair_xyzs):
                        #         print("pair", pi)
                        #         # obj_pair_xyzs: 2 * P, 5
                        #         print("collision score", collision_scores[bi, pi])
                        #         trimesh.PointCloud(obj_pair_xyz[:, :3].cpu()).show()

                        # 1 - mean() since the collision model predicts 1 if there is a collision
                        no_intersection_scores[cur_batch_idxs_start:cur_batch_idxs_end] = 1 - torch.mean(collision_scores, dim=1)
                    if visualize:
                        print("no intersection scores", no_intersection_scores)
                #######################################
                if discriminator_score_weight > 0:
                    # # debug:
                    # print(subsampled_scene_xyz.shape)
                    # print(subsampled_scene_xyz[0])
                    # trimesh.PointCloud(subsampled_scene_xyz[0, :, :3].cpu().numpy()).show()
                    #
                    with torch.no_grad():

                        # Important: since this discriminator only uses local structure param, takes sentence from the first and last position
                        # local_sentence = sentence[:, [0, 4]]
                        # local_sentence_pad_mask = sentence_pad_mask[:, [0, 4]]
                        # sentence_disc, sentence_pad_mask_disc, position_index_dic = discriminator_inference.dataset.tensorfy_sentence(raw_sentence_discriminator, raw_sentence_pad_mask_discriminator, raw_position_index_discriminator)

                        sentence_disc = torch.LongTensor([discriminator_tokenizer.tokenize(*i) for i in raw_sentence_discriminator])
                        sentence_pad_mask_disc = torch.LongTensor(raw_sentence_pad_mask_discriminator)
                        position_index_dic = torch.LongTensor(raw_position_index_discriminator)

                        preds = discriminator_model.forward(subsampled_scene_xyz,
                                                            sentence_disc.unsqueeze(0).repeat(cur_batch_size, 1).to(device),
                                                            sentence_pad_mask_disc.unsqueeze(0).repeat(cur_batch_size, 1).to(device),
                                                            position_index_dic.unsqueeze(0).repeat(cur_batch_size, 1).to(device))
                        # preds = discriminator_model.forward(subsampled_scene_xyz)
                        preds = discriminator_model.convert_logits(preds)
                        preds = preds["is_circle"]  # cur_batch_size,
                        scores[cur_batch_idxs_start:cur_batch_idxs_end] = preds
                    if visualize:
                        print("scores", scores)

            scores = scores * discriminator_score_weight + no_intersection_scores * collision_score_weight

            sort_idx = torch.argsort(scores).flip(dims=[0])[:ce_num_elite]
            elite_obj_params = sampled_obj_params[sort_idx]  # num_elite, N, 6
            elite_scores = scores[sort_idx]
            print("elite scores:", elite_scores)

            # debug: visualize elites
            # new_obj_xyzs_elites, _ = move_pc(obj_xyzs, elite_obj_params,
            #                                                            struct_pose[0:elite_obj_params.shape[0] * N],
            #                                                            current_pc_pose[0:elite_obj_params.shape[0] * N],
            #                                                            device)
            # print(new_obj_xyzs_elites.shape)
            # visualize_batch_pcs(new_obj_xyzs_elites, elite_obj_params.shape[0], N, P, limit_B=5)

            # update best so far
            # ToDo: debug this, make this faster
            for esi in range(len(elite_scores)):
                inserted = False
                for bsi in range(len(best_score_so_far)):
                    if elite_scores[esi] > best_score_so_far[bsi]:
                        best_score_so_far[bsi] = elite_scores[esi]
                        best_obj_params_so_far[bsi] = elite_obj_params[esi]
                        inserted = True
                        break
                if not inserted:
                    break
            print("best_score_so_far", best_score_so_far)
            # if elite_scores[0] > best_score_so_far:
            #     best_score_so_far = elite_scores[0]
            #     best_obj_params_so_far = elite_obj_params[0]

            sigma_eps = 0.01 + min(elite_scores) * -0.0095
            mus, sigmas = fit_gaussians(elite_obj_params.reshape(ce_num_elite, N * 6), sigma_eps)
            print("sigma noise", sigma_eps)
            print("mus", mus.shape, mus)
            print("sigmas", sigmas.shape, sigmas)
            print("sigmas avg", torch.mean(sigmas))

            avg_score = torch.mean(scores).cpu().numpy()
            print("average score of all samples in this iteration", avg_score)
            scores_average.append(avg_score)

        print("average score of all samples in all iterations", scores_average)
        print("best scores so far", best_score_so_far)
        all_scores_average.append(scores_average)

        ####################################################
        # visualize best samples
        best_obj_params_so_far = torch.cat(best_obj_params_so_far, dim=0).reshape(ce_num_best_so_far, N, 6)  # num_best_so_far, N, 6
        batch_struct_pose = struct_pose[0: ce_num_best_so_far * N]
        batch_current_pc_pose = current_pc_pose[0: ce_num_best_so_far * N]

        num_scene_pts = 4096 if discriminator_num_scene_pts is None else discriminator_num_scene_pts
        best_new_obj_xyzs, best_goal_pc_pose, best_subsampled_scene_xyz, _, _ = \
            move_pc_and_create_scene_new(obj_xyzs, best_obj_params_so_far, batch_struct_pose, batch_current_pc_pose, target_object_inds, device,
                                     return_scene_pts=True, num_scene_pts=num_scene_pts, normalize_pc=True)

        if visualize:
            visualize_batch_pcs(best_new_obj_xyzs, ce_num_best_so_far, N, P, limit_B=ce_num_best_so_far)

        # only take the best one
        # best_subsampled_scene_xyz = best_subsampled_scene_xyz[0][None, :, :]
        # best_new_obj_xyzs = best_new_obj_xyzs[0][None, :, :, :]
        # best_goal_pc_pose = best_goal_pc_pose[0][None, :, :, :]
        # best_goal_pc_pose = best_goal_pc_pose.cpu().numpy()
        # best_subsampled_scene_xyz = best_subsampled_scene_xyz.cpu().numpy()
        # best_new_obj_xyzs = best_new_obj_xyzs.cpu().numpy()
        # best_score_so_far = torch.stack(best_score_so_far).cpu().numpy()

        # take all
        best_goal_pc_pose = best_goal_pc_pose.cpu().numpy()
        best_subsampled_scene_xyz = best_subsampled_scene_xyz.cpu().numpy()
        best_new_obj_xyzs = best_new_obj_xyzs.cpu().numpy()
        best_score_so_far = torch.stack(best_score_so_far).cpu().numpy()

        ####################################################
        # verify in physics simulation
        d = {}
        d["is_circle"] = 0
        d["goal_specification"] = sample_raw_data["goal_specification"]
        d["target_objs"] = sample_raw_data["target_objs"]
        d["current_obj_poses"] = sample_raw_data["current_obj_poses"]
        d["current_pc_poses"] = sample_raw_data["current_pc_poses"]
        d["obj_perturbation_matrices"] = None

        physics_results = []
        # only evaluate the best one
        for goal_pc_pose in best_goal_pc_pose:
            d["goal_pc_poses"] = goal_pc_pose
            check, check_dict = verify_datum_in_simulation(d, assets_path, object_model_dir, structure_type=structure_type,
                                                           early_stop=physics_eval_early_stop, visualize=False)
            print("data point {}, check {}".format(data_idx, bool(check)))
            physics_results.append((check, check_dict))

        for check, check_dict in physics_results:
            print(check, check_dict)
        # visualize_batch_pcs(best_new_obj_xyzs, num_best_so_far, N, P)

        ####################################################
        # save data

        # preds = discriminator_model.forward(subsampled_scene_xyz,
        #                                     sentence.repeat(cur_batch_size, 1),
        #                                     sentence_pad_mask.repeat(cur_batch_size, 1),
        #                                     position_index.repeat(cur_batch_size, 1))

        sd = {}
        sd["json_goal_specification"] = json.dumps(sample_raw_data["goal_specification"])
        sd["json_target_objs"] = json.dumps(sample_raw_data["target_objs"])
        # a list of numpy arrays to automatically be concatenated when storing in h5
        sd["current_obj_poses"] = sample_raw_data["current_obj_poses"][:len(sample_raw_data["target_objs"])]
        # numpy
        sd["current_pc_poses"] = sample_raw_data["current_pc_poses"]
        # sd["obj_perturbation_matrices"] = None
        sd["json_sentence"] = json.dumps(sample_raw_data["sentence"])
        sd["sentence_pad_mask"] = sample_raw_data["sentence_pad_mask"]
        sd["object_pad_mask"] = sample_raw_data["object_pad_mask"]
        if discriminator_score_weight > 0:
            sd["json_raw_sentence_discriminator"] = json.dumps(raw_sentence_discriminator)
            sd["raw_sentence_pad_mask_discriminator"] = raw_sentence_pad_mask_discriminator
            sd["raw_position_index_discriminator"] = raw_position_index_discriminator

        # only save the best one
        for bsi in range(len(best_score_so_far)):

            sd["check"] = physics_results[bsi][0]
            sd["json_check_dict"] = json.dumps(convert_bool(physics_results[bsi][1]))
            sd["goal_pc_poses"] = best_goal_pc_pose[bsi]
            sd["subsampled_scene_xyz"] = best_subsampled_scene_xyz[bsi]
            sd["new_obj_xyzs"] = best_new_obj_xyzs[bsi]
            sd["discriminator_score"] = best_score_so_far[bsi]

            # for k in sd:
            #     if type(sd[k]).__module__ == np.__name__:
            #         print(k, sd[k].shape)
            #     else:
            #         print(k, sd[k])
            save_dict_to_h5(sd, filename=os.path.join(save_dir, "{}_cem{}.h5".format(sample_file_id, bsi)))

            all_eval_idxs.append(data_idx)
            if physics_results[bsi][0]:
                success_eval_idxs.append(data_idx)

        print("All:", all_eval_idxs)
        print("Success:", success_eval_idxs)

        if len(all_eval_idxs) > max_num_eval:
            break

    switch_stdout()

    return success_eval_idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval")
    parser.add_argument("--base_config_file", help='base config yaml file',
                        default='../../../configs/physics_eval/dataset_housekeep_custom/base.yaml',
                        type=str)
    parser.add_argument("--config_file", help='config yaml file',
                        default='../../../configs/physics_eval/dataset_housekeep_custom/transformer_discriminator_cem/line_test.yaml',
                        type=str)
    args = parser.parse_args()
    assert os.path.exists(args.base_config_file), "Cannot find base config yaml file at {}".format(args.config_file)
    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)
    base_cfg = OmegaConf.load(args.base_config_file)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(base_cfg, cfg)

    cfg.physics_eval_early_stop = False

    evaluate(**cfg)


    # ####################################################################################################################
    # # batch testing for testing objects
    # ####################################################################################################################
    # # dinner 10k
    # collision_model_dir = "/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220809/20220809-002824/best_model"
    # evaluate(structure_type="dinner",
    #          generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220805-153259/best_model",
    #          data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_transformer_lang_dinner_10k_test_objects_collision_100_TESTESTEST",
    #          discriminator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220809/20220813-220054_epoch46/best_model",
    #          collision_model_dir=collision_model_dir,
    #          collision_score_weight=0.0, discriminator_score_weight=1.0,
    #          ce_num_iteration=5, ce_num_samples=100, ce_num_elite=5, ce_num_best_so_far=1,
    #          discriminator_inference_batch_size=20,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=False, shuffle=False, summary_writer=None, max_num_eval=100, visualize=True,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/dinner_data/result"],
    #          override_index_dirs=["index"])
    #
    # ###################################################################################################################
    # # stacking 10k
    # evaluate(structure_type="tower",
    #          generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220809/20220815-144822/best_model",
    #          data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_transformer_lang_stacking_10k_test_objects_collision_100",
    #          discriminator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220809/20220813-220054_epoch46/best_model",
    #          collision_model_dir=collision_model_dir,
    #          collision_score_weight=0.0, discriminator_score_weight=1.0,
    #          ce_num_iteration=1, ce_num_samples=100, ce_num_elite=5, ce_num_best_so_far=1,
    #          discriminator_inference_batch_size=20,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/stacking_data/result"],
    #          override_index_dirs=["index"])
    #
    # ###################################################################################################################
    # # circle 10k
    # evaluate(structure_type="circle",
    #          generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220814-223710/best_model",
    #          data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_transformer_lang_circle_10k_test_objects_collision_100",
    #          discriminator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220809/20220813-220054_epoch46/best_model",
    #          collision_model_dir=collision_model_dir,
    #          collision_score_weight=0.0, discriminator_score_weight=1.0,
    #          ce_num_iteration=1, ce_num_samples=100, ce_num_elite=5, ce_num_best_so_far=1,
    #          discriminator_inference_batch_size=20,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/circle_data/result"],
    #          override_index_dirs=["index"])
    #
    # ###################################################################################################################
    # # line 10k
    # evaluate(structure_type="line",
    #          generator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220814-223820/best_model",
    #          data_split="train",
    #          data_root="/home/weiyu/data_drive/physics_eval_transformer_lang_line_10k_test_objects_collision_100",
    #          discriminator_model_dir="/home/weiyu/Research/intern/semantic-rearrangement/experiments/20220809/20220813-220054_epoch46/best_model",
    #          collision_model_dir=collision_model_dir,
    #          collision_score_weight=0.0, discriminator_score_weight=1.0,
    #          ce_num_iteration=1, ce_num_samples=100, ce_num_elite=5, ce_num_best_so_far=1,
    #          discriminator_inference_batch_size=20,
    #          assets_path="/home/weiyu/Research/intern/brain_gym/assets/urdf",
    #          object_model_dir="/home/weiyu/Research/intern/brain_gym/data/acronym_handpicked_v4",
    #          redirect_stdout=True, shuffle=False, summary_writer=None, max_num_eval=100, visualize=False,
    #          override_data_dirs=["/home/weiyu/data_drive/data_test_objects/line_data/result"],
    #          override_index_dirs=["index"])
    #
    # ####################################################################################################################
    # ####################################################################################################################

