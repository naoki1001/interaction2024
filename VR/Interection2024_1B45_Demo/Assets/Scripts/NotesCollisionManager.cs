using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NotesCollisionManager : MonoBehaviour
{
    public bool inLine = false;
    
    private void OnTriggerEnter(Collider collision)//接触判定
    {

        Debug.Log("接触");
        inLine = true;
    }
    private void OnTriggerExit(Collider collision)//接触終了
    {
        inLine = false;
    }
}
