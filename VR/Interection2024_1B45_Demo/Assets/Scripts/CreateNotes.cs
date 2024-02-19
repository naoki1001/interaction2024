using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CreateNotes : MonoBehaviour
{
    //Midiレシーバーで呼び出すノーツ生成用スクリプト
    public GameObject notesPrefab;
    [SerializeField, EnumIndex(typeof(JointName))] private Transform[] TapPositionAnchor;

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

    public void Note()
    {
        int index = Random.Range(0, JointNumber);
        Instantiate(notesPrefab, new Vector3(TapPositionAnchor[index].position.x, TapPositionAnchor[index].position.y, 140), transform.rotation);
    }
}
