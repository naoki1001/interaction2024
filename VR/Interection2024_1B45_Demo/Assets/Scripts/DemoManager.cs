using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DemoManager : MonoBehaviour
{
    /// <summary>
    /// ジョイント部表示プレハブ
    /// </summary>
    [SerializeField, EnumIndex(typeof(JointName))] private GameObject[] TapPositionAnchor;
    [SerializeField, EnumIndex(typeof(JointName))] private Transform[] TrackPointReferences;

    /// <summary>
    /// 各インデックスのジョイントの名前
    /// </summary>
    public enum JointName
    {
        LeftHip = 0,
        RightHip,
        LeftShoulder,
        RightShoulder,
        Head,
    }

    /// <summary>
    /// ジョイントの総数
    /// </summary>
    private int JointNumber => System.Enum.GetValues(typeof(JointName)).Length;

    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
