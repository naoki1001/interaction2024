using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NotesCollisionManager : MonoBehaviour
{
    public bool inLine = false;
    public Vector3 area;
    public GameObject note;
    
    private void OnTriggerEnter(Collider collision)//接触判定
    {

        Debug.Log("接触");
        inLine = true;
        area = collision.transform.position;
        note = collision.gameObject;
    }
    private void OnTriggerExit(Collider collision)//接触終了
    {
        inLine = false;
    }
}
