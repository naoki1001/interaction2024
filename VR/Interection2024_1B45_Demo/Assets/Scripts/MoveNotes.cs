using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveNotes : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        transform.Translate(0, 0, -130.43f * Time.deltaTime);

        if (transform.position.z < 0.0f)
        {
            Destroy(gameObject);
        }
    }
}
