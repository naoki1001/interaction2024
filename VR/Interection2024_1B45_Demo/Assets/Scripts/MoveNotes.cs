using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveNotes : MonoBehaviour
{
    public float speed = 0.50f;
    public DemoManager demoManager;
    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        transform.Translate(0, 0, -130.43f * Time.deltaTime * speed);

        if (transform.position.z < 0.0f)
        {
            demoManager.combo = 0;
            Destroy(gameObject);
        }
    }
}
