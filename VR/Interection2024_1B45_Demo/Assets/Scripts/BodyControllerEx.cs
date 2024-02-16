using System;
using System.Collections;
using System.Collections.Generic;
using Oculus.Interaction.Input;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;

public class BodyControllerEx : MonoBehaviour
{
    /// <summary>
    /// ジョイント部表示プレハブ
    /// </summary>
    [SerializeField] private GameObject TrackPointPrefab;
    [SerializeField, EnumIndex(typeof(JointName))] private Transform[] TrackPointReferences;

    // /// <summary>
    /// 各ジョイント部に配置するゲームオブジェクトの参照
    /// </summary>
    private GameObject[] TrackingPointObjects;
    /// <summary>
    /// 各インデックスのジョイントの名前
    /// </summary>
    public enum JointName
    {
        LeftHip = 0,
        RightHip,
        Hip,
        // LeftBack,
        // RightBack,
        Back, 
        LeftShoulder,
        RightShoulder,
        Neck,
        Head,
    }

    /// <summary>
    /// ジョイントの総数
    /// </summary>
    private int JointNumber => System.Enum.GetValues(typeof(JointName)).Length;
    private float H = 0.0f;
    private float y_hmd = 0.0f;

    void Start()
    {
        TrackingPointObjects = new GameObject[JointNumber];

        // 各ジョイント部のゲームオブジェクトを生成する
        for (int index = 0; index < JointNumber; index++)
        {
            var gameObject = Instantiate(TrackPointPrefab, this.transform);
            gameObject.name = ((JointName)index).ToString();
            gameObject.transform.localScale = new Vector3(0.05f, 0.05f, 0.05f);
            TrackingPointObjects[index] = gameObject;
        }
    }

    void Update()
    {
        // 各ジョイント部の座標をHead Setの値などをもとに推測する
        // Head Set
        List<InputDevice> devices = new List<InputDevice>();
        InputDevices.GetDevicesAtXRNode(XRNode.Head, devices);

        if (devices.Count > 0)
        {
            InputDevice device = devices[0];
            Vector3 position;
            Quaternion rotation;

            if (device.TryGetFeatureValue(CommonUsages.devicePosition, out position) && device.TryGetFeatureValue(CommonUsages.deviceRotation, out rotation))
            {
                H = Mathf.Max(H, position.y / 0.95f);
                y_hmd = Mathf.Max(y_hmd, position.y);
                float dev_from_mean_in_std = (H * 100.0f - 171.5f) / 6.6f;
                float diff_y = y_hmd - position.y;
                float distance = 0.10f * H;

                // z offset is 0.05f
                // ref: estat(https://www.e-stat.go.jp/dbview?sid=0003224177), https://www.jstage.jst.go.jp/article/senshoshi/63/5/63_313/_pdf

                // 頭
                TrackPointReferences[(int)JointName.Head].localPosition = position;
                TrackPointReferences[(int)JointName.Head].rotation = rotation;

                // 首
                TrackPointReferences[(int)JointName.Neck].localPosition = new Vector3(
                    0.0f,
                    dev_from_mean_in_std * 0.0614f + 1.4398f - y_hmd,
                    -0.05f
                );

                TrackPointReferences[(int)JointName.Neck].localRotation = Quaternion.Inverse(rotation);// * Quaternion.Slerp(rotation, Quaternion.identity, 0.5f);
                
                // 肩
                TrackPointReferences[(int)JointName.LeftShoulder].localPosition = new Vector3(
                    -(dev_from_mean_in_std * 0.0178f + 0.4005f) / 2,
                    dev_from_mean_in_std * 0.0489f + 1.3923f - TrackPointReferences[(int)JointName.Neck].position.y - diff_y,
                    0.0f
                );

                TrackPointReferences[(int)JointName.RightShoulder].localPosition = new Vector3(
                    (dev_from_mean_in_std * 0.0178f + 0.4005f) / 2,
                    dev_from_mean_in_std * 0.0489f + 1.3923f - TrackPointReferences[(int)JointName.Neck].position.y - diff_y,
                    0.0f
                );

                // 背中
                TrackPointReferences[(int)JointName.Back].localPosition = new Vector3(
                    0.0f,
                    dev_from_mean_in_std * 0.0454f + 1.1272f - TrackPointReferences[(int)JointName.Neck].position.y - diff_y,
                    0.0f
                );

                // TrackPointReferences[(int)JointName.LeftBack].localPosition = new Vector3(-(dev_from_mean_in_std * 0.00815f + 0.1616f), 0.0f, 0.0f);

                // TrackPointReferences[(int)JointName.RightBack].localPosition = new Vector3(dev_from_mean_in_std * 0.00815f + 0.1616f, 0.0f, 0.0f);

                // おしり
                TrackPointReferences[(int)JointName.Hip].localPosition = new Vector3(
                    0.0f,
                    dev_from_mean_in_std * 0.0368f + 0.8433f - TrackPointReferences[(int)JointName.Back].position.y - diff_y,
                    -(dev_from_mean_in_std * 0.0086f + 0.10735f)
                );

                TrackPointReferences[(int)JointName.LeftHip].localPosition = new Vector3(-(dev_from_mean_in_std * 0.0084f + 0.16375f), 0.0f, 0.0f);

                TrackPointReferences[(int)JointName.RightHip].localPosition = new Vector3(dev_from_mean_in_std * 0.0084f + 0.16375f, 0.0f, 0.0f);
            }

        }
        else
        {
            Debug.LogWarning("No device found on the Head XRNode. Check if the XR settings are correctly configured.");
        }

        // 各ジョイント部のゲームオブジェクトの座標を更新する
        for (int index = 0; index < JointNumber; index++)
        {
            if (TrackPointReferences[index] != null)
            {
                Transform posef = TrackPointReferences[index];
                TrackingPointObjects[index].transform.position = new Vector3(
                    posef.position.x,
                    posef.position.y,
                    posef.position.z
                );
                TrackingPointObjects[index].transform.rotation = new Quaternion(
                    posef.rotation.x,
                    posef.rotation.y,
                    posef.rotation.z,
                    posef.rotation.w
                );
            }
        }
    }
}