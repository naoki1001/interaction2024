using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Playables;

public class PlayAudio : MonoBehaviour
{
    public PlayableDirector director;
    private bool playing = false;
    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        if (director.time >= 4.3f && !playing)
        {
            GetComponent<AudioSource>().Play();
            playing = true;
        }
        
    }
}
