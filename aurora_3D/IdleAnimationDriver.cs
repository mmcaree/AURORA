// File: IdleAnimatorDriver.cs
using UnityEngine;

public class IdleAnimatorDriver : MonoBehaviour
{
    private Animator animator;

    void Start()
    {
        animator = GetComponent<Animator>();
    }

    void Update()
    {
        if (animator)
        {
            // Locomotion (standing still)
            animator.SetFloat("VelocityX", 0f);
            animator.SetFloat("VelocityY", 0f);
            animator.SetFloat("VelocityZ", 0f);
            animator.SetFloat("AngularY", 0f);

            animator.SetBool("Grounded", true);
            animator.SetFloat("GroundProximity", 1f); // Assume standing on ground

            animator.SetBool("Supine", false);    // Not lying down
            animator.SetFloat("Upright", 1.0f);     // Standing upright
            animator.SetBool("Seated", false);     // Not seated
            animator.SetBool("AFK", false);        // Active

            animator.SetInteger("LocomotionMode", 0); // 0 = Standing
            animator.SetBool("FootstepDisable", false); // Footsteps okay

            // Hands and Arms
            animator.SetInteger("ArmToggle", 0);
            animator.SetFloat("ArmBlendV", 0f);
            animator.SetFloat("ArmBlendH", 0f);
            animator.SetInteger("GestureLeft", 0);
            animator.SetInteger("GestureRight", 0);
            animator.SetFloat("GestureLeftWeight", 0f);
            animator.SetFloat("GestureRightWeight", 0f);
            animator.SetFloat("StandStyle", 0.8f);
            animator.SetFloat("CrouchStyle", 0f);
            animator.SetFloat("ProneStyle", 0f);

            // Face
            animator.SetInteger("ExFacial", 0); // Neutral expression
            animator.SetFloat("Voice", 0f);     // Not talking

            // Body Morphs
            animator.SetFloat("BreastSize", 0.5f); // Default
            animator.SetBool("KemonoToggle", false); // No tail/ears

            // Clothing
            animator.SetBool("HatToggle", true);
            animator.SetBool("JacketToggle", false);
            animator.SetBool("SocksToggle", false);
            animator.SetBool("ShoesToggle", false);
            animator.SetBool("UnderwearToggle", false);

            // Special Poses
            animator.SetFloat("RockNRollStyle", 0f);
        }
    }
}
