// File: AIFullBehaviorController.cs
using UnityEngine;
using System.Collections.Generic;


public class AIFullBehaviorController : MonoBehaviour
{
    public enum BehaviorState { Idle, Move, Interact }
    public enum ModeState { Work, Play, Relax }
    public enum MoodState { Positive, Neutral, Negative }
    public enum LocomotionStyle {StandStyle, CrouchStyle, ProneStyle}

    //public bool isCrouching = false;
    //public bool isProne = false;

    public ModeState currentMode = ModeState.Relax;
    public MoodState currentMood = MoodState.Neutral;
    public LocomotionStyle currentLocStyle = LocomotionStyle.StandStyle;

    //private BehaviorState currentBehavior = BehaviorState.Idle;
    private Vector3 targetPosition;


    public Animator animator;

    private float standstyle = 0.0f;

    private float idleTimer = 0f;
    private float idleTimeout = 300f; // 5 minutes = 300 seconds
    private bool isDazed = false; // Already in dazed mode?

    private float mischievousTimer = 0f;
    private float mischievousInterval = 30f; // Random interval
    
    void Start()
    {
        animator = GetComponent<Animator>();
        UpdateMood("neutral");
        animator.SetFloat("StandStyle", standstyle);
    }
    

    void Update()
    {
        HandleHotkeys();
        AutonomousBehavior();
        HandleRelaxIdleTimeout();
        HandleMischievousBehavior();
    }

    void HandleHotkeys()
    {
        if (Input.GetKeyDown(KeyCode.F1)) currentMode = ModeState.Work;
        if (Input.GetKeyDown(KeyCode.F2)) 
        {
            currentMode = ModeState.Play;
            Debug.Log("Switched to Play Mode! 😈");
        }
        if (Input.GetKeyDown(KeyCode.F3)) currentMode = ModeState.Relax;
    }

    void AutonomousBehavior()
    {
        switch (currentMode)
        {
            case ModeState.Work:
                // behave more serious
                break;
            case ModeState.Play:
                // behave playful
                break;
            case ModeState.Relax:
            default:
                // lazy slow movements
                break;
        }
    }

    void HandleMischievousBehavior()
    {
        if (currentMode != ModeState.Play)
        {
            mischievousTimer = 0f;
            return;
        }

        mischievousTimer += Time.deltaTime;

        if (mischievousTimer >= mischievousInterval)
        {
            TriggerMischievousFace();
            mischievousTimer = 0f;
            mischievousInterval = Random.Range(20f, 40f); // New random interval
        }
    }

    void TriggerMischievousFace()
    {
        if (!animator) return;

        int facial = PickRandom(new int[] {13, 15, 22}); // Mischievous faces
        animator.SetInteger("ExFacial", facial);

        Debug.Log("[Mood] Mischievous face triggered! 😈");
    }

    void HandleRelaxIdleTimeout()
    {
        if (currentMode != ModeState.Relax || currentMood != MoodState.Neutral)
        {
            idleTimer = 0f;
            isDazed = false;
            return;
        }

        idleTimer += Time.deltaTime;

        if (idleTimer >= idleTimeout && !isDazed)
        {
            ForceDazedFace();
            isDazed = true;
        }
    }

    void ForceDazedFace()
    {
        if (!animator) return;
        animator.SetInteger("ExFacial", 11); // 11 = Dazed face
    }

        // 🧠 External call from ToneReceiver
    public void UpdateMood(string tone)
    {
        switch (tone.ToLower())
        {
            case "positive":
                currentMood = MoodState.Positive;
                break;
            case "neutral":
                currentMood = MoodState.Neutral;
                break;
            case "negative":
                currentMood = MoodState.Negative;
                break;
            default:
                currentMood = MoodState.Neutral;
                break;
        }

        UpdateAnimatorMood();
    }


    void UpdateAnimatorMood()
    {
        if (!animator) return;

        int facial = 0;
        if (currentLocStyle == LocomotionStyle.StandStyle)
        {
            switch (currentMood)
            {
                //2 is wink, 16 is super sus tongue out heart eyes. May use may not use
                case MoodState.Positive:
                    facial = PickRandom(new int[] {3, 4, 5, 6, 9, 17, 18, 21});
                    if (currentMode == ModeState.Relax)
                    {
                       standstyle = PickRandom(new float[] {0.0f, 0.1f, 0.4f, 0.5f, 0.6f});
                    }

                    if (currentMode == ModeState.Play)
                    {
                       standstyle = PickRandom(new float[] {0.0f, 0.1f, 0.5f, 0.6f});
                    }
                    else
                    {
                       standstyle = PickRandom(new float[] {0.0f, 0.1f, 0.2f, 0.4f, 0.5f});
                    }
                    animator.SetFloat("StandStyle", standstyle);
                    break;
                case MoodState.Negative:
                    facial = PickRandom(new int[] {7, 8, 10, 24});
                    standstyle = PickRandom(new float[] {0.3f, 0.7f});
                    animator.SetFloat("StandStyle", standstyle);
                    break;
                case MoodState.Neutral:
                default:
                    facial = PickRandom(new int[] {1, 12, 14, 19, 20, 23});
                    if (currentMode == ModeState.Relax)
                    {
                       standstyle = PickRandom(new float[] {0.0f, 0.1f, 0.2f, 0.3f, 0.5f, 0.6f, 0.7f});
                    }

                    if (currentMode == ModeState.Play)
                    {
                       standstyle = PickRandom(new float[] {0.0f, 0.1f, 0.3f, 0.5f, 0.6f, 0.7f});
                    }
                    else
                    {
                       standstyle = PickRandom(new float[] {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.7f});
                    }
                    animator.SetFloat("StandStyle", standstyle);
                    break;
                }
            }

        animator.SetInteger("ExFacial", facial);
        Debug.Log($"[Facial] Set ExFacial to {facial} for mood {currentMood}");
    }

    T PickRandom<T>(T[] options)
    {
        return options[Random.Range(0, options.Length)];
    }
    

}

//old controller, keeping for posterity
// using UnityEngine;
// using System.Collections;

// public class AIFullBehaviorController : MonoBehaviour
// {
//     public Animator animator;
//     private bool isCrouching = false;
//     private bool isProne = false;
//     private string currentMood = "positive"; // Default

//     void Start()
//     {
//         animator = GetComponent<Animator>();
//         StartCoroutine(AIBehaviorLoop());
//     }

//     IEnumerator AIBehaviorLoop()
//     {
//         while (true)
//         {
//             yield return new WaitForSeconds(Random.Range(5f, 10f));

//             int action = Random.Range(0, 7);

//             switch (action)
//             {
//                 // case 0:
//                 //     RandomExpression();
//                 //     break;
//                 case 0:
//                     RandomClothingToggle();
//                     break;
//                 // case 2:
//                 //     RandomArmMovement();
//                 //     break;
//                 // case 3:
//                 //     RandomIdleStyle();
//                 //     break;
//                 // case 4:
//                 //     RandomKemonoToggle();
//                 //     break;
//                 // case 5:
//                 //     RandomBreastMorph();
//                 //     break;
//                 // case 4:
//                 //     RandomPosture();
//                 //     break;
//             }
//         }
//     }

//     public void SetMood(string tone)
//     {
//         currentMood = tone;
//     }

//     void RandomExpression()
//     {
//         int facial = 0;

//         if (currentMood == "positive")
//         {
//             facial = Random.Range(1, 5); // happy1 - happy3, smile
//         }
//         else if (currentMood == "negative")
//         {
//             facial = Random.Range(6, 12); // sad1, sad2, hau, etc.
//         }
//         else // neutral
//         {
//             facial = Random.Range(12, 24); // relaxed, smug, tongue move, etc.
//         }

//         animator.SetInteger("ExFacial", facial);
//     }

//     void RandomClothingToggle()
//     {
//         animator.SetBool("HatToggle", Random.value > 0.5f);
//         animator.SetBool("JacketToggle", Random.value > 0.5f);
//         animator.SetBool("SocksToggle", Random.value > 0.5f);
//         animator.SetBool("ShoesToggle", Random.value > 0.5f);
//     }

//     void RandomArmMovement()
//     {
//         animator.SetInteger("ArmToggle", Random.Range(0, 3)); // Right, Left, Both
//         animator.SetFloat("ArmBlendV", Random.Range(-1f, 1f));
//         animator.SetFloat("ArmBlendH", Random.Range(-1f, 1f));
//     }

//     void RandomIdleStyle()
//     {
//         float style = Random.Range(0f, 0.8f); // Different styles (0–0.8)
        
//         if (isProne)
//             animator.SetFloat("ProneStyle", style);
//         else if (isCrouching)
//             animator.SetFloat("CrouchStyle", style);
//         else
//             animator.SetFloat("StandStyle", style);
//     }

//     // void RandomKemonoToggle()
//     // {
//     //     animator.SetBool("KemonoToggle", Random.value > 0.7f); // Rare toggle
//     // }

//     // void RandomBreastMorph()
//     // {
//     //     animator.SetFloat("BreastSize", Random.Range(0.3f, 0.7f));
//     // }

//     void RandomPosture()
//     {
//         float chance = Random.value;

//         if (chance < 0.33f)
//         {
//             // Prone
//             isProne = true;
//             isCrouching = false;
//             animator.SetInteger("LocomotionMode", 2);
//             animator.SetFloat("Upright", 0f);
//             animator.SetFloat("ProneStyle", Random.Range(0f, 0.8f));
//             animator.SetFloat("CrouchStyle", 0.8f);
//             animator.SetFloat("StandStyle", 0.8f);
//         }
//         else if (chance < 0.66f)
//         {
//             // Crouching
//             isProne = false;
//             isCrouching = true;
//             animator.SetInteger("LocomotionMode", 1);
//             animator.SetFloat("Upright", 0.5f);
//             animator.SetFloat("CrouchStyle", Random.Range(0f, 0.8f));
//             animator.SetFloat("ProneStyle", 0.8f);
//             animator.SetFloat("StandStyle", 0.8f);
//         }
//         else
//         {
//             // Standing
//             isProne = false;
//             isCrouching = false;
//             animator.SetInteger("LocomotionMode", 0);
//             animator.SetFloat("Upright", 1f);
//             animator.SetFloat("StandStyle", Random.Range(0f, 0.8f));
//             animator.SetFloat("CrouchStyle", 0.8f);
//             animator.SetFloat("ProneStyle", 0.8f);
//         }
//     }