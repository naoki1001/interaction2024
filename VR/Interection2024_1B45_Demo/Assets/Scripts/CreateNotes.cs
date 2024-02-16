using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CreateNotes : MonoBehaviour
{
    //Midiレシーバーで呼び出すノーツ生成用スクリプト
    public GameObject notesPrefab;
    public GameObject ofukuPrefab;

    public void Note()
    {
        Instantiate(notesPrefab, new Vector3(0, 0, 140), transform.rotation);
    }
}
